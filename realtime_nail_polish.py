import argparse
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import os
# Suppress plugin loading errors - set before import
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Try to prevent plugin directory loading
if 'TF_PLUGIN_DIR' not in os.environ:
    import tempfile
    os.environ['TF_PLUGIN_DIR'] = tempfile.mkdtemp()
    
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
except Exception as e:  # pragma: no cover
    import sys
    if 'libmetal_plugin' in str(e) or 'tensorflow-plugins' in str(e):
        print("Warning: TensorFlow plugin loading issue detected.", file=sys.stderr)
        print("This shouldn't affect TFLite inference, but may cause issues.", file=sys.stderr)
        print("Consider removing system-wide TensorFlow plugins if this persists.", file=sys.stderr)
        raise SystemExit("TensorFlow import failed. Try: pip uninstall tensorflow-plugins (if installed system-wide)") from e
    raise SystemExit("TensorFlow is required. Please install it: pip install 'tensorflow>=2.12.0,<3.0.0'") from e


# ---------------- CONFIG ----------------
DEFAULT_MODEL_PATH = \
    "/Users/qari/Desktop/Farhan AI/nails_seg_s_yolov8_v1_float16.tflite"
CONF_THRESHOLD = 0.85
TEXTURE_DEFAULT = 0.35
DILATION_PIXELS = 5


# ---------------- HELPERS (adapted from provided notebook) ----------------
def adaptive_smooth_contour(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new = []
        for i in range(len(pts)):
            p_prev = pts[(i - 1) % len(pts)]
            p_curr = pts[i]
            p_next = pts[(i + 1) % len(pts)]
            smoothed = 0.5 * p_curr + 0.25 * p_prev + 0.25 * p_next
            new.append(smoothed)
        pts = np.array(new)
    return pts.astype(np.int32)


def refine_nail_mask(bin_mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    bin_mask = (bin_mask > 127).astype(np.uint8) * 255
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k_small, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_small, iterations=1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(m)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) >= 3:
            smooth = adaptive_smooth_contour(approx[:, 0, :], iterations=2)
            if len(smooth) >= 3:
                cv2.fillPoly(refined, [smooth], 255)
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    _, refined = cv2.threshold(refined, 100, 255, cv2.THRESH_BINARY)
    return refined


def parse_tflite_outputs(outputs: list) -> Tuple[Optional[int], Optional[int], list]:
    det_idx = None
    proto_idx = None
    P = None
    for i, o in enumerate(outputs):
        if o.ndim == 4 and o.shape[-1] >= 1 and o.shape[-1] <= 256:
            proto_idx = i
            P = o.shape[-1]
            break
        if o.ndim == 3 and o.shape[-1] >= 1 and o.shape[-1] <= 256 and o.shape[0] <= 1024:
            proto_idx = i
            P = o.shape[-1]
            break
    if P is not None:
        for i, o in enumerate(outputs):
            if i == proto_idx:
                continue
            if o.ndim >= 2 and o.shape[-1] == 5 + P:
                det_idx = i
                break
    if proto_idx is None:
        for i, o in enumerate(outputs):
            if o.ndim in (3, 4) and o.size > 1000:
                proto_idx = i
                P = o.shape[-1] if o.ndim in (3, 4) else None
                break
    if det_idx is None:
        for i, o in enumerate(outputs):
            if i == proto_idx:
                continue
            if o.ndim >= 2 and o.shape[-1] >= 6:
                det_idx = i
                break
    return det_idx, proto_idx, outputs


def load_tflite_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found at: {path}")
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def infer_tflite(interpreter, input_details, output_details, input_bgr: np.ndarray):
    inp_d = input_details[0]
    shape = inp_d["shape"]
    if len(shape) == 4:
        in_h, in_w = int(shape[1]), int(shape[2])
    else:
        in_h, in_w = 640, 640
    img = cv2.resize(input_bgr, (in_w, in_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if inp_d["dtype"] == np.uint8:
        input_tensor = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
    else:
        input_tensor = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(od["index"]) for od in output_details]
    return outputs, (in_w, in_h)


def overlay_results(frame_bgr: np.ndarray,
                    combined_mask: np.ndarray,
                    boxes_xyxy: np.ndarray,
                    inference_time_ms: float = 0.0,
                    total_time_ms: float = 0.0,
                    fps: float = 0.0) -> np.ndarray:
    out = frame_bgr.copy()
    # draw boxes
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if combined_mask is not None and combined_mask.sum() > 0:
        colored = np.zeros_like(out)
        colored[:, :] = (199, 21, 133)  # BGR for visualization
        alpha = (combined_mask.astype(np.float32) / 255.0) * 0.4
        alpha3 = alpha[:, :, None]
        out = (out.astype(np.float32) * (1 - alpha3) + colored.astype(np.float32) * alpha3).astype(np.uint8)
        contours, _ = cv2.findContours((combined_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 50:
                continue
            cv2.drawContours(out, [c], -1, (255, 255, 255), 1)
    
    # Draw performance metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)  # Green
    
    metrics = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_time_ms:.1f}ms",
        f"Total: {total_time_ms:.1f}ms",
    ]
    
    y_offset = 25
    for i, metric in enumerate(metrics):
        cv2.putText(out, metric, (10, y_offset + i * 25), font, font_scale, color, thickness)
    
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam inference for nail segmentation (TFLite)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .tflite model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    interpreter, input_details, output_details = load_tflite_model(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Failed to open webcam. Try a different --camera index.")

    win_name = "Nail Segmentation (TFLite, press q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Performance tracking
    frame_times = []
    max_frames_to_track = 30  # Track last 30 frames for FPS
    
    try:
        while True:
            frame_start = time.time()
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            H, W = frame_bgr.shape[:2]
            
            # Measure inference time
            inf_start = time.time()
            outputs, (in_w, in_h) = infer_tflite(interpreter, input_details, output_details, frame_bgr)
            inference_time_ms = (time.time() - inf_start) * 1000.0
            
            det_idx, proto_idx, outputs_all = parse_tflite_outputs(outputs)

            combined_mask = np.zeros((H, W), dtype=np.uint8)
            boxes_xyxy = []

            if proto_idx is not None and det_idx is not None:
                proto = outputs_all[proto_idx]
                if proto.ndim == 4 and proto.shape[0] == 1:
                    proto = proto[0]
                elif proto.ndim == 3 and proto.shape[0] == 1:
                    proto = proto[0]
                if proto.ndim != 3:
                    proto = None

                det = outputs_all[det_idx]
                if det.ndim == 3 and det.shape[0] == 1:
                    det = det[0]
                if det.ndim == 2 and det.shape[0] < det.shape[1] and det.shape[0] <= 50:
                    det = det.transpose(1, 0)

                if proto is not None and det.ndim == 2:
                    P = proto.shape[-1]
                    num_cols = det.shape[1]
                    if num_cols >= 5 + P:
                        mask_coeffs = det[:, -P:]
                        scores = det[:, 4]
                        boxes = det[:, :4]
                    else:
                        mask_coeffs = det[:, -P:]
                        scores = det[:, 4] if det.shape[1] > 4 else np.zeros(det.shape[0])
                        boxes = det[:, :4] if det.shape[1] >= 4 else np.zeros((det.shape[0], 4))

                    valid = scores > float(args.conf)
                    boxes = boxes[valid]
                    mask_coeffs = mask_coeffs[valid]

                    ph, pw = proto.shape[0], proto.shape[1]
                    for i_det in range(len(mask_coeffs)):
                        coeff = mask_coeffs[i_det]
                        mask = np.tensordot(proto, coeff, axes=([2], [0]))
                        mask = 1.0 / (1.0 + np.exp(-mask))
                        mask_in = cv2.resize((mask * 255.0).astype(np.uint8), (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                        cx, cy, bw, bh = boxes[i_det]
                        if cx > 1.5 or cy > 1.5 or bw > 1.5 or bh > 1.5:
                            x1 = int(cx - bw / 2)
                            y1 = int(cy - bh / 2)
                            x2 = int(cx + bw / 2)
                            y2 = int(cy + bh / 2)
                        else:
                            x1 = int((cx - bw / 2) * in_w)
                            y1 = int((cy - bh / 2) * in_h)
                            x2 = int((cx + bw / 2) * in_w)
                            y2 = int((cy + bh / 2) * in_h)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(in_w, x2), min(in_h, y2)
                        full_mask_in = np.zeros((in_h, in_w), dtype=np.uint8)
                        if x2 > x1 and y2 > y1:
                            region = mask_in[y1:y2, x1:x2]
                            if region.size == 0:
                                full_mask_in[:, :] = mask_in
                            else:
                                full_mask_in[y1:y2, x1:x2] = region
                        else:
                            full_mask_in[:, :] = mask_in
                        mask_resized = cv2.resize(full_mask_in, (W, H), interpolation=cv2.INTER_LINEAR)
                        _, m_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
                        combined_mask = cv2.bitwise_or(combined_mask, m_bin)
                        boxes_xyxy.append([max(0, x1 * W // in_w), max(0, y1 * H // in_h),
                                           min(W - 1, x2 * W // in_w), min(H - 1, y2 * H // in_h)])
            else:
                # Fallback: search for per-instance masks tensors
                found_masks = False
                masks_tensor = None
                for o in outputs_all:
                    if proto_idx is not None and o is outputs_all[proto_idx]:
                        continue
                    if det_idx is not None and o is outputs_all[det_idx]:
                        continue
                    if o.ndim == 4 and o.shape[0] == 1:
                        cand = o[0]
                        if cand.ndim == 3 and cand.shape[0] <= 200:
                            masks_tensor = cand
                            found_masks = True
                            break
                    if o.ndim == 3 and o.shape[0] <= 200 and o.shape[1] > 1 and o.shape[2] > 1:
                        masks_tensor = o
                        found_masks = True
                        break
                if found_masks and masks_tensor is not None:
                    for i in range(masks_tensor.shape[0]):
                        m = masks_tensor[i]
                        m_resized = cv2.resize((m * 255.0).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
                        _, m_bin = cv2.threshold(m_resized, 127, 255, cv2.THRESH_BINARY)
                        combined_mask = cv2.bitwise_or(combined_mask, m_bin)

            # refine and slight dilation to stabilize mask visualization
            if combined_mask.sum() > 0:
                refined_mask = refine_nail_mask(combined_mask, min_area=100)
                kernel = np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8)
                refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
            else:
                refined_mask = combined_mask

            # Calculate FPS and total processing time
            total_time_ms = (time.time() - frame_start) * 1000.0
            frame_times.append(total_time_ms)
            if len(frame_times) > max_frames_to_track:
                frame_times.pop(0)
            avg_frame_time_ms = np.mean(frame_times) if frame_times else total_time_ms
            fps = 1000.0 / avg_frame_time_ms if avg_frame_time_ms > 0 else 0.0

            vis = overlay_results(frame_bgr, refined_mask, np.array(boxes_xyxy) if boxes_xyxy else np.zeros((0, 4)),
                                 inference_time_ms=inference_time_ms, total_time_ms=total_time_ms, fps=fps)
            cv2.imshow(win_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


