# Crack Detection on the Edge

**A YOLOv8n crack detector compressed from 6.25 MB / 38.6 ms → 3.23 MB / 4.93 ms with no measurable accuracy loss, deployed on a Raspberry Pi 5 mounted on a VOLTA Bot Sync.**

| | FP32 baseline | Final (`S2_QAT_INT8 @ 416 px`) |
|---|---|---|
| mAP@0.5 | 0.7146 | **0.7114** (−0.0032) |
| CPU latency (p50) | 38.60 ms | **4.93 ms** |
| Throughput | 25.9 FPS | **202.8 FPS** |
| File size | 6.25 MB | **3.23 MB** |
| | | **7.83× faster · 1.93× smaller** |

The deployed artefact is [`edge_ai_final_2/edge_ai_final_2/S2_QAT_INT8_r416.tflite`](edge_ai_final_2/edge_ai_final_2/S2_QAT_INT8_r416.tflite) (3.23 MB).

**Authors:** Priyanshi Dubey, Niyati Jawariya
**Course:** [Edge AI 2026](https://www.samy101.com/edge-ai-26/)

---

## Why this exists

Bridges, dams, tunnels, and building facades all develop cracks over time. Inspecting them today still means a human with a clipboard, or a rope-access team scaling a structure — slow, expensive, inconsistent, and dangerous.

A mobile robot with a camera and an **on-device** detector can scan large surfaces continuously and flag defects in real time. The hard part is fitting a useful detector inside the constraints:

- **No GPU, no guaranteed cloud** — the bot only carries a Pi 5.
- **Real-time or bust** — slow inference forces the bot to crawl.
- **Cracks are thin, low-contrast targets** — an off-the-shelf COCO detector won't do.

So we trained YOLOv8n on the BD3 building-defect dataset, then took it through a five-stage compression pipeline (PTQ → QAT → pruning → resolution sweep) to land at a 3.23 MB INT8 TFLite that runs the live RealSense feed in real time on the Pi 5.

---
a
## Quickstart — inference on the Pi

```python
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(
    model_path='S2_QAT_INT8_r416.tflite',
    num_threads=4,
)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# capture from RealSense → letterbox to 416×416 → quantise → invoke → NMS
interpreter.set_tensor(input_details[0]['index'], img_int8)
interpreter.invoke()
boxes = interpreter.get_tensor(output_details[0]['index'])
```

Runtime: TFLite Python interpreter with the **XNNPACK delegate** and `num_threads=4`. Benchmarks use 50 warmup runs followed by 200 timed runs at batch size 1.

---

## What's in this repo

```
crack-detection-edge/
├── edge_ai_final_2/edge_ai_final_2/
│   ├── S2_QAT_INT8_r416.tflite              ← deployed model (3.23 MB)
│   └── quantize_yolov8n_final_graphs.ipynb  ← end-to-end pipeline + plots
└── README.md
```

The notebook is the end-to-end source of truth — it runs all five optimisation stages and produces every plot and number quoted below.

---

## Hardware & software

**Hardware on the bot**

| Component | Role |
|---|---|
| VOLTA Bot Sync | Autonomous mobile platform |
| Raspberry Pi 5 | On-board inference |
| Intel RealSense D455 | RGB-D capture (depth used downstream for crack sizing) |
| Joystick controller | Manual override during data capture and supervised navigation |

**Software stack**

Python 3.10+ · Ultralytics YOLO ≥ 8.2 · PyTorch ≥ 2.1 · TensorFlow / TFLite ≥ 2.15 · ONNX ≥ 1.15 · OpenCV ≥ 4.8 · Roboflow

---

## Dataset

Trained on the [BD3 building-defect dataset](https://github.com/Praveenkottari/BD3-Dataset) ([Kaggle mirror](https://www.kaggle.com/datasets/praveenkottari/bd3-dataset-for-building-defect-detection)).

- 1,745 source images → **4,189 after Roboflow augmentation** (flips, 90° rotations, ±11° rotation, ±13°/±14° shear, 0–28% crop, up to 2.4 px blur — all chosen to mirror real bot-eye conditions).
- Splits: **3,666 train / 349 val / 174 test**.
- Polygon annotations converted to axis-aligned boxes for YOLO training.

---

## Approach in five stages

We didn't pick a compression strategy upfront — we ran the candidates and let the numbers decide.

| Stage | Pipeline | Technique |
|---|---|---|
| 0 | `S0_FP32` | FP32 baseline |
| 1 | `S1_FP16`, `S1_INT8` | Post-training quantisation |
| 2 | `S2_QAT_INT8` | Quantisation-aware training (snap-to-INT8-grid callback after every batch) |
| 3 | `S3_PrunedINT8` | 10% L1-unstructured prune → masked fine-tune → INT8 |
| 4 | sweep | Top-2 pipelines × {640, 512, 416, 320} px |

Pareto front after stages 1–3: `S1_INT8`, `S2_QAT_INT8`, `S3_PrunedINT8`. The top-2 by PiScore (`S3_PrunedINT8` and `S2_QAT_INT8`) advanced to the resolution sweep.

### PiScore — how the winner is chosen

Final model selection is multi-objective. **PiScore** combines accuracy, latency, size, and accuracy-loss into a single auditable score in [0, 1]:

| Metric | Weight | Direction |
|---|---|---|
| `mAP@0.5` | 0.40 | higher is better |
| `p50_cpu_ms` | 0.35 | lower is better |
| `size_mb` | 0.15 | lower is better |
| `mAP_drop` (vs FP32) | 0.10 | lower is better |

Each metric is min–max normalised across the candidate pool, then weighted and summed. For the Stage 4 resolution sweep the weights shift to 0.7 / 0.3 (accuracy / latency) since size barely moves with image size.

---

## Why YOLOv8n (and why it quantises cleanly)

YOLOv8n beat YOLOv8s and YOLOv10n on **accuracy, size, and CPU latency simultaneously** on this dataset (mAP@0.5 = 0.714 at 5.97 MB / 3.12 ms). Cracks are thin and low-feature; the larger v8s backbone over-fits and loses recall.

It also quantises well because:

- Standard Conv-BN-SiLU blocks throughout — every layer has a battle-tested INT8 TFLite kernel.
- C2f modules give predictable, bounded activation ranges.
- Decoupled detection head — classification and regression branches quantise independently with no shared logits.
- Anchor-free design — fewer post-processing ops to quantise; NMS runs in float on CPU after dequantisation.
- SiLU is monotonic and bounded; TFLite implements it via INT8 lookup table.
- Only 3.0 M parameters, so quantisation error has fewer layers to compound through.

In practice: <0.5 mAP loss under PTQ INT8, fully recovered with QAT.

---

## Results — final model

**Winner: `S2_QAT_INT8 @ 416 px`** — QAT-trained, INT8-quantised YOLOv8n re-exported and validated at 416×416.

| | Value |
|---|---|
| mAP@0.5 | 0.7114 |
| mAP@0.5:0.95 | 0.5258 |
| CPU p50 | 4.93 ms |
| Throughput | 202.8 FPS |
| File size | 3.23 MB |
| Speed-up vs FP32 | 7.83× |
| Compression vs FP32 | 1.93× |
| mAP drop vs FP32 | 0.0032 |

At 4.93 ms per frame on a 4-thread CPU, the model leaves headroom for camera capture, pre-processing, and post-processing on the Pi 5 while still hitting real-time framerates for live navigation.

The full per-stage / per-resolution breakdown, training curves, Pareto plots, and PiScore bars are produced by the notebook.

---

## Deployment

- **Camera:** RealSense D455 at 1280×720, letter-boxed and resized to 416×416.
- **Inference:** `tflite_runtime` with XNNPACK, 4 threads.
- **Post-processing:** detections projected back to the original frame; depth values from the D455 used downstream to estimate physical crack extent.
- **Bot integration:** detection results published over a ROS topic on the VOLTA Bot Sync. Joystick override available for closer inspection of flagged surfaces.

---

## Reproducing the pipeline

1. Pull the BD3 dataset (link above) and re-export from Roboflow with the augmentation settings listed in the **Dataset** section.
2. Open [`edge_ai_final_2/edge_ai_final_2/quantize_yolov8n_final_graphs.ipynb`](edge_ai_final_2/edge_ai_final_2/quantize_yolov8n_final_graphs.ipynb) and run top-to-bottom — it produces every artefact (S0–S4), every plot, and the final TFLite model.

---

## Future work

- **Structured pruning** — channel-level pruning would let TFLite shrink conv shapes for real on-device speed gains (unlike unstructured sparsity, which TFLite stores densely).
- **Dedicated accelerators** — Coral USB TPU and Hailo-8 are next on the bench; both offer 5–10× further INT8 latency reductions.
- **Crack severity grading** — use D455 depth to estimate physical crack width and classify hairline / minor / major.
- **Continual learning** — periodically retrain on field captures the bot itself collects, to cover surface types and lighting absent from BD3.

---

## References

1. P. Kottari, *BD3: Building Defect Detection Dataset.* [github.com/Praveenkottari/BD3-Dataset](https://github.com/Praveenkottari/BD3-Dataset)
2. Ultralytics, *YOLOv8 documentation.* [docs.ultralytics.com](https://docs.ultralytics.com)
3. A. Wang et al., *YOLOv10: Real-Time End-to-End Object Detection.* [github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
4. TensorFlow, *Post-training quantization.* [tensorflow.org/lite/performance/post_training_quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
5. PyTorch, *Pruning tutorial.* [pytorch.org/tutorials/intermediate/pruning_tutorial.html](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
