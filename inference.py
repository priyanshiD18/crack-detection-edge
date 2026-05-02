"""
Real-time crack detection on Raspberry Pi (INT8 QAT model).
 
- Live preview window with bounding boxes
- Saves a snapshot when crack boxes cover >= MIN_AREA_PCT of the frame
  (rate-limited so the same crack doesn't get saved 100x in a row)
- Optionally records the full session to an .mp4
- CSV log of every snapshot
 
"""
import cv2
import numpy as np
import time
import csv
from datetime import datetime
from pathlib import Path
 
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter
 
# -------------------- CONFIG --------------------
MODEL_PATH            = "/home/winterschool1/Downloads/edge_project/qat_int8_TRUE_r416.tflite"      #add the path to your model here
IMG_SIZE              = 416
CONF_THRESH           = 0.47     
IOU_THRESH            = 0.45
FRAME_SKIP            = 2
CAM_W, CAM_H          = 640, 480
CAM_INDEX_OVERRIDE    = 4
 
# Outputs
OUT_DIR               = Path("/home/winterschool1/Downloads/edge_project/captures")    #add the path to your desired output dir here
SNAPSHOT_DIR          = OUT_DIR / "snapshots"
RECORD_VIDEO          = False
SNAPSHOT_COOLDOWN_SEC = 1.5
MIN_AREA_PCT          = 1.5
 
# False-positive filters (per-box, after NMS)
MIN_BOX_AREA_PCT      = 0.15      # drop tiny corner-noise boxes
MAX_BOX_AREA_PCT      = 65.0      # drop "whole-wall" boxes
MIN_ASPECT_RATIO      = 1.2      # cracks are elongated; drop near-square boxes
                                  # AR = max(w,h)/min(w,h);
 
# CAMERA INDEX 
def find_working_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            continue
        print(f"camera found at index {i}")
        return i
    raise RuntimeError("no working RGB camera found")
 
if CAM_INDEX_OVERRIDE is not None:
    CAM_INDEX = CAM_INDEX_OVERRIDE
    print(f"using forced camera index = {CAM_INDEX}")
else:
    CAM_INDEX = find_working_camera()
 
# LOAD MODEL
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp  = interpreter.get_input_details()[0]
outp = interpreter.get_output_details()[0]
in_idx, out_idx     = inp['index'], outp['index']
in_dtype, out_dtype = inp['dtype'], outp['dtype']
in_scale,  in_zp    = inp['quantization']
out_scale, out_zp   = outp['quantization']
print(f"input  {inp['shape']}  {in_dtype.__name__}  q=({in_scale},{in_zp})")
print(f"output {outp['shape']} {out_dtype.__name__}  q=({out_scale},{out_zp})")
 
# CAMERA SETUP
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("camera not working")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
 
# OUTPUT DIRS + LOG + VIDEO WRITER 
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path    = OUT_DIR / f"detections_{session_tag}.csv"
log_file    = open(log_path, "w", newline="")
log_writer  = csv.writer(log_file)
log_writer.writerow(["timestamp", "snapshot_file", "num_boxes",
                     "max_conf", "area_pct"])
 
video_writer = None
video_path   = None
if RECORD_VIDEO:
    video_path = OUT_DIR / f"session_{session_tag}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (CAM_W, CAM_H))
    print(f"recording video -> {video_path}")
print(f"snapshots dir   -> {SNAPSHOT_DIR}")
print(f"detection log   -> {log_path}")
 
#  PREPROCESS (INT8-aware + resize + color convert) 
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if in_dtype == np.uint8:
        # INT8 model: feed raw uint8 RGB; the input quant scale (1/255)
        # internally maps it to 0..1 float, which is what the network expects.
        return np.expand_dims(img, axis=0).astype(np.uint8)
    elif in_dtype == np.int8:
        # Symmetric int8: shift uint8 [0..255] -> int8 [-128..127]
        return np.expand_dims(img.astype(np.int16) - 128, axis=0).astype(np.int8)
    else:
        # FP32 fallback
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
 
# POSTPROCESS (INT8-aware + coord fix + filters) 
def postprocess(frame, raw):
    """Returns (annotated_frame, num_boxes, max_conf, area_pct)."""
    arr = np.squeeze(raw)
 
    # 1. Dequantize if model output is integer.
    if out_dtype in (np.uint8, np.int8):
        arr = (arr.astype(np.float32) - out_zp) * out_scale
 
    if arr.ndim != 2 or arr.shape[0] != 5:
        return frame, 0, 0.0, 0.0
    cx, cy, ww, hh, conf = arr
 
    # 2. Auto-scale normalized 0..1 coords up to IMG_SIZE pixels.
    if float(np.max(np.abs(cx))) <= 1.5 and float(np.max(np.abs(cy))) <= 1.5:
        cx = cx * IMG_SIZE
        cy = cy * IMG_SIZE
        ww = ww * IMG_SIZE
        hh = hh * IMG_SIZE
 
    # 3. Confidence gate.
    keep = conf >= CONF_THRESH
    if not np.any(keep):
        return frame, 0, 0.0, 0.0
    cx, cy, ww, hh, conf = cx[keep], cy[keep], ww[keep], hh[keep], conf[keep]
    x1, y1 = cx - ww / 2, cy - hh / 2
    x2, y2 = cx + ww / 2, cy + hh / 2
 
    # 4. NMS in input-pixel space.
    boxes_xywh = np.stack([x1, y1, ww, hh], axis=1).astype(np.float32)
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        conf.astype(np.float32).tolist(),
        CONF_THRESH, IOU_THRESH,
    )
    if len(idxs) == 0:
        return frame, 0, 0.0, 0.0
 
    H, W = frame.shape[:2]
    sx, sy = W / IMG_SIZE, H / IMG_SIZE
    frame_area = float(H * W)
 
    # 5. Per-box plausibility filter (size + aspect ratio).
    survivors = []
    for i in np.array(idxs).flatten():
        bw_px = (x2[i] - x1[i]) * sx
        bh_px = (y2[i] - y1[i]) * sy
        if bw_px <= 0 or bh_px <= 0:
            continue
        box_pct = (bw_px * bh_px) / frame_area * 100.0
        ar      = max(bw_px, bh_px) / max(min(bw_px, bh_px), 1.0)
        if box_pct < MIN_BOX_AREA_PCT or box_pct > MAX_BOX_AREA_PCT:
            continue
        if ar < MIN_ASPECT_RATIO:
            continue
        survivors.append(i)
 
    if not survivors:
        return frame, 0, 0.0, 0.0
 
    # 6. Draw + accumulate mask for true area % (overlap counts once).
    mask = np.zeros((H, W), dtype=np.uint8)
    n = 0
    max_conf = 0.0
    for i in survivors:
        bx1 = int(max(0, x1[i] * sx))
        by1 = int(max(0, y1[i] * sy))
        bx2 = int(min(W, x2[i] * sx))
        by2 = int(min(H, y2[i] * sy))
        cv2.rectangle(mask,  (bx1, by1), (bx2, by2), 255, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, f"crack {conf[i]:.2f}", (bx1, max(0, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        n += 1
        if conf[i] > max_conf:
            max_conf = float(conf[i])
 
    area_pct = float(np.count_nonzero(mask)) / frame_area * 100.0
    return frame, n, max_conf, area_pct
 
#  MAIN LOOP 
print("starting (ESC in window to quit)")
frame_count     = 0
fps_smooth      = 0.0
last_snapshot_t = 0.0
total_snapshots = 0
 
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame not received, retrying...")
            continue
        frame_count += 1
 
        if frame_count % FRAME_SKIP != 0:
            if video_writer is not None:
                video_writer.write(frame)
            cv2.imshow("Crack Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
 
        x = preprocess(frame)
        interpreter.set_tensor(in_idx, x)
        t0 = time.time()
        interpreter.invoke()
        dt = time.time() - t0
        raw = interpreter.get_tensor(out_idx)
 
        frame, n_boxes, max_conf, area_pct = postprocess(frame, raw)
 
        fps = 1.0 / max(dt, 1e-6)
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
        status = (f"FPS: {fps_smooth:.1f}  inf: {dt*1000:.0f}ms  "
                  f"snaps: {total_snapshots}  area: {area_pct:.1f}%")
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 
        now = time.time()
        if (n_boxes > 0
                and area_pct >= MIN_AREA_PCT
                and (now - last_snapshot_t) >= SNAPSHOT_COOLDOWN_SEC):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            snap_name = (f"crack_{ts}_n{n_boxes}"
                         f"_c{int(max_conf*100):02d}"
                         f"_a{int(area_pct*10):03d}.jpg")
            snap_path = SNAPSHOT_DIR / snap_name
            cv2.imwrite(str(snap_path), frame)
            log_writer.writerow([ts, snap_name, n_boxes,
                                 f"{max_conf:.4f}", f"{area_pct:.2f}"])
            log_file.flush()
            last_snapshot_t = now
            total_snapshots += 1
            cv2.circle(frame, (CAM_W - 20, 20), 8, (0, 0, 255), -1)
 
        if video_writer is not None:
            video_writer.write(frame)
        cv2.imshow("Crack Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
 
finally:
    cap.release()
    if video_writer is not None:
        video_writer.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"\nexited. {total_snapshots} snapshots saved to {SNAPSHOT_DIR}")
    print(f"detection log: {log_path}")
    if RECORD_VIDEO and video_path is not None:
        print(f"session video: {video_path}")
 
MODEL_PATH            = "/home/winterschool1/Downloads/edge_project/qat_int8_TRUE_r416.tflite"
IMG_SIZE              = 416
CONF_THRESH           = 0.47
IOU_THRESH            = 0.45
FRAME_SKIP            = 2
CAM_W, CAM_H          = 640, 480
CAM_INDEX_OVERRIDE    = 4
 
# Outputs
OUT_DIR               = Path("/home/winterschool1/Downloads/edge_project/captures")
SNAPSHOT_DIR          = OUT_DIR / "snapshots"
RECORD_VIDEO          = False
SNAPSHOT_COOLDOWN_SEC = 1.5
MIN_AREA_PCT          = 1.5
 
# Per-box plausibility filters
MIN_BOX_AREA_PCT      = 0.15
MAX_BOX_AREA_PCT      = 65.0
MIN_ASPECT_RATIO      = 1.2
 
# TEMPORAL TRACKING (to reduce flicker + false positives)
TRACK_IOU_THRESH      = 0.3       # how much overlap counts as "same crack"
MIN_HITS              = 4         # need this many consecutive matches to confirm
MAX_MISS              = 3         # drop a track if it disappears for this many frames
SHOW_PENDING          = True      # draw unconfirmed boxes faintly (debug aid)
 