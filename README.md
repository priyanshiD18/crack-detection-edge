# crack-detection-edge
Real-time crack detection on Raspberry Pi 5 using quantized YOLOv8n. Compares FP32 (baseline), PTQ-FP16, PTQ-INT8, QAT-INT8, and pruned-INT8 variants, followed by a resolution sweep on the top-2 picks across mAP, latency, and size. Deploys QAT-INT8 TFLite (~3.4 MB, ~18 ms) with live RealSense RGB inference and Volta bot sync.
