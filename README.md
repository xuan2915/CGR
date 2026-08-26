# BVR: Training-Free Boundary-Verified Refinement

BVR is a training-free two-pass framework for video temporal grounding. It
first grounds the query on the full video, constructs a compact temporal crop
from the top-k prediction cluster, and then refines the event boundaries
inside that crop with a second inference pass.

The method requires no training or adapter updates. It is evaluated on the
QVHighlights validation set with a frozen multimodal large language model.

## Requirements

- Python 3.10+
- PyTorch and Transformers
- The MLLM codebase and checkpoint expected by the inference script
- QVHighlights data and videos under the default local paths

## Usage

Run BVR inference:

```bash
python scripts/infer_bvr.py \
  --pred_path outputs/bvr_top5_pad025 \
  --model_gnd_path /path/to/mllm_checkpoint \
  --crop_mode top5_cluster \
  --pad_ratio 0.25 \
  --device cuda:0
```

Evaluate the generated predictions:

```bash
python scripts/evaluate.py \
  outputs/bvr_top5_pad025/output.jsonl \
  data/qvhighlights/highlight_val_release.jsonl
```

## Main Results

Results on the QVHighlights validation set (1,550 video-query pairs):

| Method | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
| --- | ---: | ---: | ---: | ---: |
| 64f uniform | 78.90 | 65.55 | 44.71 | 57.97 |
| 96f uniform | 81.35 | 68.32 | 49.74 | 61.16 |
| Naive crop-refine | 80.13 | 67.10 | 48.00 | 59.73 |
| **BVR (ours)** | **81.29** | **69.03** | **50.00** | **61.35** |

## License

This project is released under the BSD-3-Clause License.

## Acknowledgement

This work builds on the QVHighlights dataset.
