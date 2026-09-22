# CGR: Training-Free Cluster-Guided Refinement for Video Temporal Grounding

<p align="center">
  <a href="https://github.com/xuan2915/CGR"><img src="https://img.shields.io/badge/GitHub-CGR-blue"></a>
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-green">
  <img src="https://img.shields.io/badge/Under%20Review-orange">
</p>

> **TL;DR:** CGR is a training-free refinement procedure for video temporal grounding. It first grounds the query on the full video, clusters the top-k predicted spans into one crop window, and then re-grounds the query inside that window at a denser effective frame rate. It needs no training, no model modification and no extra peak GPU memory.

---

## 📌 Overview

**CGR** is a lightweight refinement procedure built on top of [VideoMind](https://github.com/yeliudev/VideoMind). It targets a practical problem in multimodal large language model (MLLM) grounding:

- Uniform global sampling cannot provide enough temporal detail near event boundaries.
- Uniformly dense sampling raises peak GPU memory roughly in proportion to the frame count.
- Existing coarse-to-fine refinement can dilute the effective frame rate when the crop window is too wide, which is what happens when the window is built from an unfiltered set of candidates.

CGR addresses these limitations with a training-free two-pass design:

1. **Pass 1: Global coarse grounding.** The frozen MLLM localizes the query on the full video with 64 frames at 1.0 fps and returns 60 ranked candidate spans.
2. **Cluster construction.** The top-k candidates are merged into a single temporal cluster spanning from their minimum start to their maximum end.
3. **Crop window construction.** The cluster is padded by a ratio alpha on both sides and clamped to the video, giving the crop window W.
4. **Pass 2: Dense re-grounding.** The identical frozen MLLM re-grounds the query inside W with 64 frames decoded at 2.0 fps.
5. **Prediction merging.** The coarse and refined candidates are concatenated, sorted by confidence, and truncated to 100; the top-1 becomes the prediction.

The two-pass construction is the one used by existing zoom-in refinement; what CGR contributes is the rule that decides the window, `k=5` with `alpha=0.25` plus the skip gate below, together with the measurement that justifies it. The point of the measurement is that the window is not a free parameter: it is set by how far apart the model's coarse candidates are, and on QVHighlights that makes it far wider than a compact window. Getting it wrong costs the whole second pass.

Windows shorter than 2 s and windows covering at least 95% of the video are skipped, in which case the Pass-1 result is returned unchanged. The effective sampling rate inside W is `F / |W|`, which saturates at the 2.0 fps decode rate whenever `|W| < 32` s.

---

## 🔑 Key Design

| Component | Description |
|---|---|
| Global grounding | Full video, 64 frames at 1.0 fps |
| Candidate set | 60 spans from the regression head, sorted by confidence |
| Prediction cluster | Top-k spans merged as `[min start, max end]` |
| Padding ratio | alpha = 0.25 by default |
| Crop window | Cluster padded by alpha, clamped to the video |
| Refinement pass | 64 frames decoded inside W at 2.0 fps |
| Prediction merge | Concatenate coarse and refined candidates, sort by confidence, keep top 100 |
| Effective rate | `F / |W|`, i.e. 0.79 fps at the measured median window of 81 s |

The operating point `k=5`, `alpha=0.25` balances boundary coverage against effective frame-rate dilution. Widening the cluster beyond that point keeps growing the window without improving accuracy, and narrowing it to `k=1` starts to clip the true event boundary.

---

## 🏗️ Repository Structure

```text
CGR/
├── scripts/
│   ├── infer_cgr.py        # Two-pass CGR inference
│   ├── evaluate.py         # QVHighlights metric evaluation
│   └── audit_crop.py       # Measures |W| and the refinement skip rate
├── outputs/
│   └── cgr_top5_pad025.jsonl   # Predictions of the default setting
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📦 Installation

### Prerequisites

- Python >= 3.10
- CUDA 11.8 or newer on NVIDIA GPUs
- PyTorch >= 2.0
- QVHighlights videos and annotations

### 1. Clone this repository

```bash
git clone https://github.com/xuan2915/CGR.git
cd CGR
```

### 2. Prepare the base MLLM grounding environment

The inference script imports the `videomind` package from [VideoMind](https://github.com/yeliudev/VideoMind). Install that codebase according to its own instructions first:

```bash
git clone https://github.com/yeliudev/VideoMind.git
cd VideoMind
pip install -r requirements.txt
```

Return to this repository and install the remaining requirements:

```bash
cd ../CGR
pip install -r requirements.txt
```

### 3. Download the 7B checkpoint

Download the grounding checkpoint used by the base codebase and store it in `model_zoo/`:

```bash
mkdir -p model_zoo
huggingface-cli download yeliudev/VideoMind-7B --local-dir model_zoo/VideoMind-7B
```

### 4. Prepare QVHighlights

Place the QVHighlights validation annotations at:

```text
data/qvhighlights/highlight_val_release.jsonl
```

Place the videos at:

```text
data/qvhighlights/videos_3fps_480_noaudio/
```

The annotation file is expected to contain `vid`, `qid`, `query`, and `relevant_windows` fields.

---

## 🚀 Quick Start

### Run CGR inference

```bash
python scripts/infer_cgr.py \
  --pred_path outputs/cgr_top5_pad025 \
  --model_gnd_path model_zoo/VideoMind-7B \
  --crop_mode top5_cluster \
  --pad_ratio 0.25 \
  --num_threads 1 \
  --device cuda:0
```

The output file is written to:

```text
outputs/cgr_top5_pad025/output.jsonl
```

Each line is a JSON object:

```json
{"vid": "video_id", "qid": 0, "pred_relevant_windows": [[start, end, confidence], ...]}
```

The line also carries the crop geometry of that sample, so that the window and the effective rate can be audited later:

```json
{"refine_ran": true, "top_n": 5, "pad_ratio": 0.25,
 "cluster_start": 90.0, "cluster_end": 116.0, "cluster_span": 26.0,
 "crop_start": 83.5, "crop_end": 122.5, "crop_dur": 39.0, "fps_eff": 1.64}
```

### Key arguments

| Argument | Description |
|---|---|
| `--pred_path` | Output directory for the prediction JSONL file |
| `--model_gnd_path` | Path to the grounding MLLM checkpoint |
| `--crop_mode` | Number of coarse predictions used to build the cluster |
| `--pad_ratio` | Padding added to both sides of the cluster |
| `--num_threads` | Video decoding threads |
| `--device` | Inference device, e.g. `cuda:0` |

---

## 🔮 Evaluation

Evaluate the generated predictions:

```bash
python scripts/evaluate.py \
  outputs/cgr_top5_pad025.jsonl \
  data/qvhighlights/highlight_val_release.jsonl
```

The evaluator reports R1, R5 and mAP at IoU 0.3 / 0.5 / 0.7 plus mIoU. The bundled prediction file reproduces the paper's default-setting row exactly:

```text
MR-full-R1@0.3: 81.29
MR-full-R1@0.5: 69.03
MR-full-R1@0.7: 50.00
MR-full-mIoU:   61.35
```

### Measure the crop window

`|W|` depends only on Pass-1 predictions, so it can be measured with a single forward pass instead of a full two-pass run:

```bash
python scripts/audit_crop.py \
  --pred_path outputs/crop_audit_top5_pad025 \
  --model_gnd_path model_zoo/VideoMind-7B \
  --crop_mode top5_cluster \
  --pad_ratio 0.25
```

The script prints the mean and median `|W|`, the median effective rate and the fraction of queries whose refinement was skipped, and writes one JSON line per sample to `crop_audit.jsonl`.

---

## ⚙️ Hyperparameters

| Argument | Values | Default | Description |
|---|---|---|---|
| `--crop_mode` | `top1_cluster`, `top3_cluster`, `top5_cluster`, `top10_cluster` | `top5_cluster` | Number of coarse predictions used to build the temporal cluster |
| `--pad_ratio` | float | `0.25` | Padding added to both sides of the cluster |
| `--num_threads` | int | `1` | Video decoding threads |
| `--device` | `auto`, `cuda:N`, `cpu` | `auto` | Inference device |

---

## 📊 Main Results

Results on the QVHighlights validation set (1,550 video-query pairs). Every configuration runs the same frozen VideoMind-7B grounder and differs only in which frames it sees: 64 frames in a single pass is the model as published, 96 frames in a single pass is the direct way to buy temporal resolution, and CGR is two passes of 64 frames.

| Configuration | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| VideoMind 64f, 1 pass | 78.90 | 65.55 | 44.71 | 57.97 | 1x |
| VideoMind 96f, 1 pass | 81.35 | 68.32 | 49.74 | 61.16 | 1.5x |
| **CGR, 2 passes of 64f** | **81.29** | **69.03** | **50.00** | **61.35** | **1x** |

Paired bootstrap over the 1,550 queries (10,000 resamples, 95% percentile interval) and McNemar's test on the R1@0.7 hits. "conservative" is the `k=10, alpha=0.50` crop-window configuration of the ablation below.

| Comparison | Metric | Difference | 95% CI | p |
| --- | --- | ---: | --- | ---: |
| CGR − VideoMind 64f | R1@0.7 | +5.29 | [+3.55, +7.10] | <0.001 |
| CGR − VideoMind 64f | mIoU | +3.39 | [+2.51, +4.29] | <0.001 |
| CGR − conservative | R1@0.7 | +2.00 | [+0.26, +3.81] | 0.016 |
| CGR − conservative | mIoU | +1.62 | [+0.74, +2.53] | <0.001 |
| CGR − VideoMind 96f | R1@0.7 | +0.26 | [−2.00, +2.45] | 0.42 |
| CGR − VideoMind 96f | mIoU | +0.20 | [−0.98, +1.39] | 0.37 |
| VideoMind 96f − 64f | R1@0.7 | +5.03 | [+2.77, +7.35] | <0.001 |
| VideoMind 96f − 64f | mIoU | +3.19 | [+2.02, +4.37] | <0.001 |

The first four rows are the significant gains; the middle two are the "no detectable difference" group, which is what the memory claim rests on; the last two are the yardstick for how large a real gain looks on this benchmark.

CGR is significantly better than the model's own single pass and than the conservative crop window, and statistically indistinguishable from a 96-frame forward pass while running at single-pass peak memory. The price is a second forward pass: CGR spends inference time to buy back memory.

### Ablation: the crop window

CGR keeps the two-pass construction and changes only how the window is built, so the ablation varies that construction and nothing else.

| Crop window | R1@0.7 | mIoU |
| --- | ---: | ---: |
| conservative, k=10, alpha=0.50 | 48.00 | 59.73 |
| CGR, k=1, alpha=0.25 | 47.10 | 58.31 |
| CGR, k=3, alpha=0.25 | 49.48 | 60.78 |
| **CGR, k=5, alpha=0.25** | **50.00** | **61.35** |
| CGR, k=3, alpha=0.15 | 47.68 | 60.30 |
| CGR, k=3, alpha=0.35 | 48.58 | 60.24 |

The conservative corner is the natural first guess, since a large cluster and generous padding both reduce the risk of excluding the true event. It is also the worst of the tuned settings: the second pass gains only 3.29 points R1@0.7 over the plain single pass, against 5.29 points once the window is built by CGR, and it lands 1.74 points below the 96-frame budget while still costing a second pass. A conservative window is therefore not a safe window, and its size has to be measured rather than guessed.

### Ablation: Pass-1 frame budget (k=5, alpha=0.25)

| Pass-1 frames | R1@0.7 | mIoU |
| --- | ---: | ---: |
| 48 | 44.84 | 58.27 |
| **64** | **50.00** | **61.35** |

Coarse Pass-1 predictions from too few frames are unreliable, the cluster inherits those errors, and the refinement pass cannot recover boundaries that fall outside a misplaced window.

### Measured crop-window geometry

Medians over the 1,550 validation queries, recovered from the crop geometry each run logged; alpha=0.25 unless stated. A skipped refinement is counted as a window of the full video length, so the effective rate `F / |W|` is defined for every row, and the two uniform baselines perform no refinement at all.

| Strategy | Window (s) | % of video | Effective FPS | Skipped | R1@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| VideoMind 64f, 1 pass | 150 | 100 | 0.43 | – | 44.71 |
| VideoMind 96f, 1 pass | 150 | 100 | 0.64 | – | 49.74 |
| conservative (k=10, alpha=0.50) | 130 | 87 | 0.49 | 43.7% | 48.00 |
| CGR (k=1) | 42 | 28 | 1.52 | 2.9% | 47.10 |
| CGR (k=3) | 68 | 45 | 0.94 | 14.5% | 49.48 |
| **CGR (k=5)** | **81** | **54** | **0.79** | **20.6%** | **50.00** |
| CGR (k=3, alpha=0.15) | 60 | 40 | 1.07 | 10.8% | 47.68 |
| CGR (k=3, alpha=0.35) | 73 | 49 | 0.88 | 17.1% | 48.58 |

Reading the last two columns together gives the trade-off: enlarging the window lowers the effective rate while accuracy first rises and then falls, so the best configuration is an interior one rather than the widest.

The window is wider than the cluster itself because a couple of low-confidence candidates that agree with neither the top-5 nor each other can stretch the cluster across most of the video; this is why the conservative top-10 setting refines 87% of the video and is slower than the 96-frame budget. The spread within a single configuration is larger than the difference between configurations: at the operating point the 5th to 95th percentile of `|W|` runs from 33 s to the full video.

---

## 📌 Reproducibility Notes

- Both passes process 64 frames, resized to the 36x28x28 to 64x28x28 pixel budget.
- Pass 1 samples at 1.0 fps over the full video; Pass 2 decodes at 2.0 fps inside the crop window and then uniformly subsamples to the 64-frame budget.
- Decoding is greedy with 256 maximum output tokens.
- Predictions are rounded to the QVHighlights 2.0-second annotation unit.
- The example prediction file in `outputs/cgr_top5_pad025.jsonl` reproduces the paper's default-setting metrics with `scripts/evaluate.py` and can be evaluated without a GPU.

---

## 📖 Citation

If you find this work helpful, please cite our paper:

```bibtex
@misc{cgr2026,
  title={CGR: Training-Free Cluster-Guided Refinement for Video Temporal Grounding},
  author={Anonymous Authors},
  year={2026}
}
```

---

## 📜 License

This project is released under the [BSD-3-Clause License](LICENSE).

## 🙏 Acknowledgement

We thank the QVHighlights dataset and the upstream [VideoMind](https://github.com/yeliudev/VideoMind) project for providing the grounding codebase and checkpoints used in this work.
