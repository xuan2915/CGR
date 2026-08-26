# BVR: Training-Free Boundary-Verified Refinement for Video Temporal Grounding

<p align="center">
  <a href="https://github.com/xuan2915/BVR"><img src="https://img.shields.io/badge/GitHub-BVR-blue"></a>
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-green">
  <img src="https://img.shields.io/badge/Under%20Review-orange">
</p>

> **TL;DR:** BVR is a training-free two-pass framework for video temporal grounding. It first coarsely grounds the query on the full video, constructs a compact temporal crop from the top-k prediction cluster, and then re-grounds the query inside that crop with a second inference pass to refine event boundaries.

---

## Overview

BVR addresses the frame-resolution tradeoff in multimodal large language model (MLLM) temporal grounding. Uniform global sampling cannot provide enough temporal detail near event boundaries, while uniformly dense sampling is often infeasible under a fixed GPU memory budget.

BVR keeps the peak memory cost of a single inference pass and restores temporal resolution only around predicted event regions:

1. **Pass 1: Global coarse grounding.** The frozen MLLM localizes the query on the full video with a coarse sampling rate.
2. **Temporal cluster construction.** The top-*k* predictions are merged into one compact temporal cluster.
3. **Crop window construction.** The cluster is expanded by a padding ratio alpha to retain boundary coverage.
4. **Pass 2: Dense boundary refinement.** The same frozen MLLM re-grounds the query on the cropped segment at a higher effective frame rate.
5. **Prediction merging.** Pass 1 and Pass 2 predictions are merged and sorted by confidence.

The method requires no training, no model modification, and no additional peak GPU memory beyond single-pass inference.

---

## Key Design

| Component | Description |
|---|---|
| Global grounding | Full-video sampling at 1.0 fps with 64 frames |
| Prediction cluster | Top-*k* predictions merged by min start and max end |
| Padding ratio | alpha = 0.25 by default |
| Refinement pass | Crop-window sampling at 2.0 fps with 64 frames |
| Prediction merge | Concatenate coarse and refined candidates, sort by confidence |
| Effective FPS | Approx. 1.82 in the default 35-second refinement window |

The default setting `k=5`, `alpha=0.25` balances boundary coverage against effective frame-rate dilution.

---

## Repository Structure

```text
BVR/
??? scripts/
?   ??? infer_bvr.py        # Two-pass BVR inference
?   ??? evaluate.py         # QVHighlights metric evaluation
??? outputs/
?   ??? bvr_top5_pad025.jsonl  # Example predictions
??? requirements.txt
??? LICENSE
??? README.md
```

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU with at least 45 GB memory for the 7B model
- PyTorch >= 2.0
- QVHighlights videos and annotations

### 1. Clone this repository

```bash
git clone https://github.com/xuan2915/BVR.git
cd BVR
```

### 2. Prepare the base MLLM grounding environment

The inference script imports the `videomind` package from the base MLLM grounding codebase. Install that codebase according to its own instructions first:

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

## Usage

### Run BVR inference

```bash
python scripts/infer_bvr.py \
  --pred_path outputs/bvr_top5_pad025 \
  --model_gnd_path model_zoo/VideoMind-7B \
  --crop_mode top5_cluster \
  --pad_ratio 0.25 \
  --num_threads 1 \
  --device cuda:0
```

The output file is written to:

```text
outputs/bvr_top5_pad025/output.jsonl
```

Each line is a JSON object:

```json
{"vid": "video_id", "qid": 0, "pred_relevant_windows": [[start, end, confidence], ...]}
```

### Evaluate the predictions

```bash
python scripts/evaluate.py \
  outputs/bvr_top5_pad025/output.jsonl \
  data/qvhighlights/highlight_val_release.jsonl
```

The evaluator prints `R1@0.3`, `R1@0.5`, `R1@0.7`, `R5@0.3`, `R5@0.5`, `R5@0.7`, `mAP@0.3`, `mAP@0.5`, `mAP@0.7`, and `mIoU`.

---

## Hyperparameters

| Argument | Values | Default | Description |
|---|---|---|---|
| `--crop_mode` | `top1_cluster`, `top3_cluster`, `top5_cluster`, `top10_cluster` | `top3_cluster` | Number of coarse predictions used to build the temporal cluster |
| `--pad_ratio` | float | `0.25` | Padding added to both sides of the cluster |
| `--num_threads` | int | `1` | Video decoding threads |
| `--device` | `auto`, `cuda:N`, `cpu` | `auto` | Inference device |

The paper uses `top5_cluster` and `pad_ratio=0.25`. BVR skips the refinement pass when the crop window is too short or covers more than 95% of the video.

---

## Main Results

Results on the QVHighlights validation set (1,550 video-query pairs):

| Method | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
| --- | ---: | ---: | ---: | ---: |
| 64f uniform | 78.90 | 65.55 | 44.71 | 57.97 |
| 96f uniform | 81.35 | 68.32 | 49.74 | 61.16 |
| Naive crop-refine | 80.13 | 67.10 | 48.00 | 59.73 |
| **BVR (ours)** | **81.29** | **69.03** | **50.00** | **61.35** |

### Effective frame rate

| Strategy | Window | Window ratio | Effective FPS |
| --- | ---: | ---: | ---: |
| 64f uniform | 150 s | 100% | 0.43 |
| 96f uniform | 150 s | 100% | 0.64 |
| Naive crop-refine | ~70 s | 47% | 0.91 |
| **BVR** | ~35 s | 24% | **1.82** |

---

## Reproducibility Notes

- Both passes process 64 frames at 36x28x28 to 64x28x28 pixels.
- Pass 1 samples at 1.0 fps on the full video.
- Pass 2 samples at 2.0 fps inside the crop window.
- Greedy decoding is used with 256 maximum output tokens.
- Predictions are rounded to the QVHighlights 2.0-second annotation unit.
- The example prediction file in `outputs/bvr_top5_pad025.jsonl` can be evaluated directly with `scripts/evaluate.py`.

---

## License

This project is released under the [BSD-3-Clause License](LICENSE).

## Acknowledgement

We thank the QVHighlights dataset and the upstream VideoMind project for providing the grounding codebase and checkpoints used in this work.
