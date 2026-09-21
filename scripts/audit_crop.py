"""Measure CGR's crop window without running the (expensive) refinement pass.

The crop window W depends ONLY on Pass-1 predictions, so |W|, the cluster span
|T|, the effective fps and the refinement skip-rate can all be measured with a
single forward pass per sample -- half the cost of a full CGR run.  This is the
quantity behind the crop-window geometry table of the paper.  The saved
prediction files do not record W, and it cannot be reconstructed from them,
because Pass-1 and Pass-2 predictions are indistinguishable after the
confidence merge.

Pass 1 here uses byte-for-byte the same prompt, frame budget, sampling rate and
rounding as scripts/infer_cgr.py, so the crop geometry it reports is the crop
geometry CGR would actually use.

Quick smoke test on just the figure's cases:
    python scripts/audit_crop.py --vids 74jtgDnsEBU_210.0_360.0,uoVRb7a58GU_210.0_360.0
Full audit over all 1,550 validation samples:
    python scripts/audit_crop.py --pred_path /path/to/outputs/crop_audit_top5_pad025
"""

import argparse
import json
import os

import nncore
import torch
from videomind.constants import GROUNDER_PROMPT
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.parser import parse_span

ANNO_PATH = "data/qvhighlights/highlight_val_release.jsonl"
VIDEO_ROOT = "data/qvhighlights/videos_3fps_480_noaudio"
UNIT = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", required=True, help="directory to write crop_audit.jsonl")
    p.add_argument("--model_gnd_path", required=True)
    p.add_argument("--crop_mode", default="top5_cluster")
    p.add_argument("--pad_ratio", type=float, default=0.25)
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="only the first N samples (0 = all)")
    p.add_argument("--vids", default="", help="comma-separated vid list (overrides --limit)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples already present in the output file")
    return p.parse_args()


def load_annos(want):
    annos = []
    with open(ANNO_PATH, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if want and d["vid"] not in want:
                continue
            annos.append(dict(
                vid=d["vid"], qid=d["qid"], query=d["query"],
                duration=float(d.get("duration", 150.0)),
                video_path=os.path.join(VIDEO_ROOT, d["vid"] + ".mp4")))
    return annos


def main():
    args = parse_args()
    top_n = {"top1_cluster": 1, "top3_cluster": 3, "top5_cluster": 5,
             "top10_cluster": 10}.get(args.crop_mode, 5)
    print(f"crop audit: crop_mode={args.crop_mode} (k={top_n}) pad_ratio={args.pad_ratio}")

    want = set(v.strip() for v in args.vids.split(",") if v.strip())
    annos = load_annos(want)
    if args.limit and not want:
        annos = annos[:args.limit]
    if not annos:
        raise SystemExit("no matching samples")
    print(f"loaded {len(annos)} samples")

    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    os.makedirs(args.pred_path, exist_ok=True)
    out_path = os.path.join(args.pred_path, "crop_audit.jsonl")

    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["vid"], r["qid"]))
        print(f"resuming: {len(done)} samples already done")
    elif not args.resume and os.path.exists(out_path):
        # fresh run requested -> move the old file aside instead of clobbering it
        os.rename(out_path, out_path + ".prev")
        print(f"existing {out_path} renamed to {out_path}.prev")

    todo = [a for a in annos if (a["vid"], a["qid"]) not in done]
    print(f"{len(todo)} samples to process")
    # Append + flush per sample so an interrupted run still leaves usable data
    # (nncore.dump writes everything only at the very end).
    fh = open(out_path, "a", encoding="utf-8")
    for i in nncore.ProgressBar(range(len(todo))):
        anno = todo[i]
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": anno["video_path"],
             "num_threads": args.num_threads,
             "min_pixels": 36 * 28 * 28, "max_pixels": 64 * 28 * 28,
             "max_frames": 64, "fps": 1.0},
            {"type": "text", "text": GROUNDER_PROMPT.format(anno["query"])}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        img, vid = process_vision_info(msgs)
        data = processor(text=[text], images=img, videos=vid, return_tensors="pt")
        data = {k: v.to(device) for k, v in data.items()}
        model.set_adapter("grounder")
        model.generate(**data, do_sample=False, max_new_tokens=256)

        duration = anno["duration"]
        if len(model.reg) == 0:
            pred_full = torch.tensor([[0.0, duration, 1.0]])
        else:
            blob = model.reg[0].cpu().float()
            pred, conf = blob[:, :2] * duration, blob[:, 2:]
            pred = pred.clamp(0, duration)
            pred = torch.round(pred / UNIT).long() * UNIT
            bad = pred[:, 1] - pred[:, 0] < 0
            pred[bad] = pred[bad].roll(1, dims=1)
            pred_full = torch.cat([pred, conf], dim=1)

        # ---- crop geometry: exactly the logic of scripts/infer_cgr.py -------
        top = pred_full[:top_n]
        cs, ce = float(top[:, 0].min()), float(top[:, 1].max())
        span = ce - cs
        crop_s, crop_e = parse_span([cs - args.pad_ratio * span,
                                     ce + args.pad_ratio * span], duration)
        crop_dur = crop_e - crop_s
        refine_ran = bool(crop_dur > 2.0 and crop_dur < duration * 0.95)

        row = dict(
            vid=anno["vid"], qid=anno["qid"],
            pass1_preds=pred_full.tolist(),
            crop_mode=args.crop_mode, top_n=top_n, pad_ratio=args.pad_ratio,
            cluster_start=cs, cluster_end=ce, cluster_span=span,
            crop_start=crop_s, crop_end=crop_e, crop_dur=crop_dur,
            fps_eff=64.0 / max(crop_dur, 1e-6), refine_ran=refine_ran,
            skip_reason=None if refine_ran else (
                "too_short" if crop_dur <= 2.0 else "covers_almost_whole_video"))
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"{i}: {anno['vid']} |T|={span:6.1f}s W=[{crop_s:6.1f},{crop_e:6.1f}] "
              f"|W|={crop_dur:6.1f}s fps={64.0/max(crop_dur,1e-6):5.2f} "
              f"{'refine' if refine_ran else 'SKIP'}")
    fh.close()

    # summarise whatever is in the file now (works after a resume too)
    rows = []
    with open(out_path, encoding="utf-8") as f2:
        for line in f2:
            if line.strip():
                rows.append(json.loads(line))

    W = [r["crop_dur"] for r in rows]
    S = [r["cluster_span"] for r in rows]
    ran = [r["refine_ran"] for r in rows]
    Ws = sorted(W)
    med = Ws[len(Ws) // 2]
    print(f"\nwrote {out_path}  ({len(rows)} samples)")
    print(f"  |W|  mean={sum(W)/len(W):.1f}s  median={med:.1f}s  ({100*med/150:.0f}% of video)")
    print(f"  |T|  mean={sum(S)/len(S):.1f}s  median={sorted(S)[len(S)//2]:.1f}s")
    print(f"  median FPS_eff = {64.0/med:.2f}")
    print(f"  refinement skipped: {sum(1 for x in ran if not x)}/{len(ran)}"
          f" ({100*sum(1 for x in ran if not x)/len(ran):.1f}%)")
    print("\nNow copy crop_audit.jsonl back to your workstation for analysis.")


if __name__ == "__main__":
    main()
