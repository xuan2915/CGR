# CGR: Training-Free Cluster-Guided Refinement for Video Temporal Grounding

<p align="center">
  <a href="https://github.com/xuan2915/BVR"><img src="https://img.shields.io/badge/GitHub-CGR-blue"></a>
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-green">
  <img src="https://img.shields.io/badge/Under%20Review-orange">
</p>

> **TL;DR:** CGR is a training-free two-pass framework for video temporal grounding. It first grounds the query on the full video, clusters the top-k predicted spans into one crop window, and then re-grounds the query inside that window at a denser effective frame rate. It needs no training, no model modification and no extra peak GPU memory.

---

## 📌 Overview

**CGR** is a lightweight temporal grounding framework built on top of [VideoMind](https://github.com/yeliudev/VideoMind). It targets a practical problem in multimodal large language model (MLLM) grounding:

- Uniform global sampling cannot provide enough temporal detail near event boundaries.
- Uniformly dense sampling raises peak GPU memory roughly in proportion to the frame count.
- Existing coarse-to-fine refinement can dilute the effective frame rate when the crop window is too wide, which is what happens when the window is built from an unfiltered set of candidates.

CGR addresses these limitations with a training-free two-pass design:

1. **Pass 1: Global coarse grounding.** The frozen MLLM localizes the query on the full video with 64 frames at 1.0 fps and returns 60 ranked candidate spans.
2. **Cluster construction.** The top-k candidates are merged into a single temporal cluster spanning from their minimum start to their maximum end.
3. **Crop window construction.** The cluster is padded by a ratio alpha on both sides and clamped to the video, giving the crop window W.
4. **Pass 2: Dense re-grounding.** The identical frozen MLLM re-grounds the query inside W with 64 frames decoded at 2.0 fps.
5. **Prediction merging.** The coarse and refined candidates are concatenated, sorted by confidence, and truncated to 100; the top-1 becomes the prediction.

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

The default setting `k=5`, `alpha=0.25` balances boundary coverage against effective frame-rate dilution. Widening the cluster beyond that point keeps growing the window without improving accuracy, and narrowing it to `k=1` starts to clip the true event boundary.

---

## 🏗️ Repository Structure

```text
BVR/
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
git clone https://github.com/xuan2915/BVR.git
cd BVR
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
cd ../BVR
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

Results on the QVHighlights validation set (1,550 video-query pairs). All four configurations are evaluated under identical conditions.

| Method | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
| --- | ---: | ---: | ---: | ---: |
| 64f uniform | 78.90 | 65.55 | 44.71 | 57.97 |
| 96f uniform | 81.35 | 68.32 | 49.74 | 61.16 |
| Naive crop-refine (k=10, alpha=0.50) | 80.13 | 67.10 | 48.00 | 59.73 |
| **CGR (ours, k=5, alpha=0.25)** | **81.29** | **69.03** | **50.00** | **61.35** |

Paired bootstrap over the 1,550 queries (10,000 resamples, 95% percentile interval) and McNemar's test on the R1@0.7 hits:

| Comparison | Metric | Difference | 95% CI | p |
| --- | --- | ---: | --- | ---: |
| CGR − 64f | R1@0.7 | +5.29 | [+3.55, +7.10] | <0.001 |
| CGR − naive | R1@0.7 | +2.00 | [+0.26, +3.81] | 0.016 |
| CGR − 96f | R1@0.7 | +0.26 | [−2.00, +2.45] | 0.42 |
| 96f − 64f | R1@0.7 | +5.03 | [+2.77, +7.35] | <0.001 |

CGR is significantly better than the 64-frame baseline and than naive crop-refine, and statistically indistinguishable from dense 96-frame uniform sampling while running at single-pass peak memory.

### Ablation: cluster size and padding

| Setting | R1@0.7 | mIoU |
| --- | ---: | ---: |
| k=1, alpha=0.25 | 47.10 | 58.31 |
| k=3, alpha=0.25 | 49.48 | 60.78 |
| **k=5, alpha=0.25** | **50.00** | **61.35** |
| k=3, alpha=0.15 | 47.68 | 60.30 |
| k=3, alpha=0.35 | 48.58 | 60.24 |

### Ablation: Pass-1 frame budget (k=5, alpha=0.25)

| Pass-1 frames | R1@0.7 | mIoU |
| --- | ---: | ---: |
| 48 | 44.84 | 58.27 |
| **64** | **50.00** | **61.35** |

Coarse Pass-1 predictions from too few frames are unreliable, the cluster inherits those errors, and the refinement pass cannot recover boundaries that fall outside a misplaced window.

### Measured crop-window geometry

Medians over the 1,550 validation queries, recovered from the crop geometry each run logged. A skipped refinement is counted as a window of the full video length, so the effective rate `F / |W|` is defined identically for every row.

| Strategy | Window (s) | Window (%) | Effective FPS | Skipped |
| --- | ---: | ---: | ---: | ---: |
| 64f uniform | 150 | 100% | 0.43 | – |
| 96f uniform | 150 | 100% | 0.64 | – |
| Naive (top10, alpha=0.50) | 130 | 87% | 0.49 | 43.7% |
| CGR (top1, alpha=0.25) | 42 | 28% | 1.52 | 2.9% |
| CGR (top3, alpha=0.25) | 68 | 45% | 0.94 | 14.5% |
| **CGR (top5, alpha=0.25)** | **81** | **54%** | **0.79** | **20.6%** |
| CGR (top3, alpha=0.15) | 60 | 40% | 1.07 | 10.8% |
| CGR (top3, alpha=0.35) | 73 | 49% | 0.88 | 17.1% |

The window is wider than the cluster itself because one low-confidence candidate can stretch the cluster across most of the video; this is why the naive top-10 setting refines 87% of the video and is slower than dense 96-frame sampling. Accuracy peaks at the intermediate windows around the default setting.

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
