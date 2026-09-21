"""CGR standalone inference - bypasses qvhighlights.py completely.

Two-pass, training-free, cluster-guided refinement for video temporal grounding.

    Pass 1  ground the query on the full video with 64 frames at 1.0 fps
    Cluster take the top-k Pass-1 spans, cluster them as [min start, max end]
    Window  pad the cluster by a factor alpha, clamped to the video
    Pass 2  re-ground the query inside the window with 64 frames at 2.0 fps
    Merge   concatenate both candidate sets, sort by confidence, keep top 100

Refinement is skipped when the window would be shorter than 2 s or would
cover at least 95% of the video.
"""
import argparse, copy, json
import nncore, torch
from videomind.constants import GROUNDER_PROMPT
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration
from videomind.utils.parser import parse_span


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", required=True)
    p.add_argument("--model_gnd_path", required=True)
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--crop_mode", default="top5_cluster",
                   choices=["top1_cluster", "top3_cluster", "top5_cluster",
                            "top10_cluster"],
                   help="how many Pass-1 candidates form the cluster")
    p.add_argument("--pad_ratio", type=float, default=0.25,
                   help="padding added to both sides of the cluster")
    return p.parse_args()


def load_annos():
    path = "data/qvhighlights/highlight_val_release.jsonl"
    annos = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                annos.append(dict(
                    vid=d["vid"], qid=d["qid"], query=d["query"],
                    duration=d.get("duration", 150.0),
                    video_path=f"data/qvhighlights/videos_3fps_480_noaudio/{d['vid']}.mp4",
                    span=d.get("relevant_windows", [])))
    return annos


def main():
    args = parse_args()
    print(f"CGR standalone: crop={args.crop_mode} pad={args.pad_ratio}")
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device
    annos = load_annos()
    print(f"Loaded {len(annos)} samples")
    unit = 2.0
    import os; os.makedirs(args.pred_path, exist_ok=True)
    pred_path = f"{args.pred_path}/output.jsonl"

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = annos[i]; dump = {}
        video_path, query, duration = anno["video_path"], anno["query"], anno["duration"]
        print(f"\n{i}: {anno['vid']} dur={int(duration)}s q={query[:50]}")

        # Pass 1: 64 frames over the full video at 1.0 fps
        msgs = [{"role":"user","content":[
            {"type":"video","video":video_path,"num_threads":args.num_threads,
             "min_pixels":36*28*28,"max_pixels":64*28*28,"max_frames":64,"fps":1.0},
            {"type":"text","text":GROUNDER_PROMPT.format(query)}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        img, vid = process_vision_info(msgs)
        data = processor(text=[text], images=img, videos=vid, return_tensors="pt")
        data = {k:v.to(device) for k,v in data.items()}
        model.set_adapter("grounder"); model.generate(**data, do_sample=False, max_new_tokens=256)
        if len(model.reg) == 0:
            pred_full = torch.tensor([[0, duration, 1.0]])
        else:
            blob = model.reg[0].cpu().float()
            pred, conf = blob[:,:2]*duration, blob[:,2:]
            pred = pred.clamp(0, duration)
            pred = torch.round(pred/unit).long()*unit
            bad = pred[:,1]-pred[:,0] < 0; pred[bad] = pred[bad].roll(1, dims=1)
            pred_full = torch.cat([pred, conf], dim=1)
        print(f"P1 top1: {pred_full[0].tolist()}")

        # keep the Pass-1 candidate set and the crop geometry: the merged
        # prediction list alone does not record which window was refined
        meta = dict(pass1_preds=pred_full.tolist())

        # Pass 2: dense re-grounding inside the crop window
        top_n = {"top1_cluster":1,"top3_cluster":3,"top5_cluster":5,"top10_cluster":10}.get(args.crop_mode,5)
        top_preds = pred_full[:top_n]
        cs, ce = float(top_preds[:,0].min()), float(top_preds[:,1].max())
        span = ce - cs
        crop_s, crop_e = parse_span([cs-args.pad_ratio*span, ce+args.pad_ratio*span], duration)
        crop_dur = crop_e - crop_s
        if crop_dur > 2.0 and crop_dur < duration*0.95:
            print(f"CGR crop: [{crop_s:.1f},{crop_e:.1f}]")
            cmsgs = [{"role":"user","content":[
                {"type":"video","video":video_path,"num_threads":args.num_threads,
                 "video_start":crop_s,"video_end":crop_e,
                 "min_pixels":36*28*28,"max_pixels":64*28*28,"max_frames":64,"fps":2.0},
                {"type":"text","text":GROUNDER_PROMPT.format(query)}]}]
            ctext = processor.apply_chat_template(cmsgs, add_generation_prompt=True)
            cimg, cvid = process_vision_info(cmsgs)
            cdata = processor(text=[ctext], images=cimg, videos=cvid, return_tensors="pt")
            cdata = {k:v.to(device) for k,v in cdata.items()}
            model.set_adapter("grounder"); model.generate(**cdata, do_sample=False, max_new_tokens=256)
            if len(model.reg) > 0:
                rblob = model.reg[0].cpu().float()
                rpred = rblob[:,:2]*crop_dur + crop_s
                rpred = rpred.clamp(0, duration)
                rpred = torch.round(rpred/unit).long()*unit
                rbad = rpred[:,1]-rpred[:,0] < 0; rpred[rbad] = rpred[rbad].roll(1,dims=1)
                merged = torch.cat([pred_full[:,:2], rpred]); mconf = torch.cat([pred_full[:,2:], rblob[:,2:]])
                merged = torch.cat([merged, mconf], dim=1)
                _, idx = merged[:,2].sort(descending=True)
                pred_full = merged[idx][:100]
                print(f"CGR top1: {pred_full[0].tolist()}")
                # record crop geometry + raw Pass-2 output
                meta.update(dict(
                    refine_ran=True, crop_mode=args.crop_mode, top_n=top_n,
                    pad_ratio=args.pad_ratio, cluster_start=cs, cluster_end=ce,
                    cluster_span=span, crop_start=crop_s, crop_end=crop_e,
                    crop_dur=crop_dur, fps_eff=64.0 / max(crop_dur, 1e-6),
                    pass2_preds_crop=rblob[:, :2].tolist(),
                    pass2_conf=rblob[:, 2:].flatten().tolist(),
                    pass2_preds_global=rpred.tolist()))
        else:
            print(f"CGR skip: crop_dur={crop_dur:.1f}s")
            # record why refinement was skipped
            meta.update(dict(
                refine_ran=False, crop_mode=args.crop_mode, top_n=top_n,
                pad_ratio=args.pad_ratio, cluster_start=cs, cluster_end=ce,
                cluster_span=span, crop_start=crop_s, crop_end=crop_e,
                crop_dur=crop_dur, fps_eff=64.0 / max(crop_dur, 1e-6),
                skip_reason=("too_short" if crop_dur <= 2.0
                             else "covers_almost_whole_video")))
        print("CGR_META " + json.dumps({k: v for k, v in meta.items()
                                        if k != "pass1_preds"}))
        dump["vid"]=anno["vid"]; dump["qid"]=anno["qid"]
        dump["pred_relevant_windows"]=pred_full.tolist()
        dump.update(meta)
        dumps.append(dump)
    nncore.dump(dumps, pred_path); print(f"\nDone: {pred_path}")


if __name__ == "__main__": main()
