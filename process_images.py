import argparse
import os
import cv2
import numpy as np

# Suppress plugin loading errors - set before import
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if 'TF_PLUGIN_DIR' not in os.environ:
    import tempfile
    os.environ['TF_PLUGIN_DIR'] = tempfile.mkdtemp()
    
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
except Exception as e:
    import sys
    if 'libmetal_plugin' in str(e) or 'tensorflow-plugins' in str(e):
        print("Warning: TensorFlow plugin loading issue detected.", file=sys.stderr)
        raise SystemExit("TensorFlow import failed. Try: pip uninstall tensorflow-plugins (if installed system-wide)") from e
    raise SystemExit("TensorFlow is required. Please install it: pip install 'tensorflow>=2.12.0,<3.0.0'") from e

# ---------------- CONFIG ----------------
DEFAULT_MODEL_PATH = "/Users/qari/Desktop/Farhan AI/nails_seg_s_yolov8_v1_float16.tflite"
CONF_THRESHOLD = 0.85
MIN_CONTOUR = 100
DILATION_PIXELS = 5

# ---------------- HELPERS ----------------
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

def parse_tflite_outputs(outputs: list):
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

def overlay_mask_on_image(original_bgr: np.ndarray, mask: np.ndarray, color: tuple = (199, 21, 133), alpha: float = 0.5) -> np.ndarray:
    """Overlay mask on original image with translucent color. No boxes or text."""
    result = original_bgr.copy()
    if mask is not None and mask.sum() > 0:
        colored_overlay = np.zeros_like(result)
        colored_overlay[:, :] = color  # BGR color (magenta)
        mask_alpha = (mask.astype(np.float32) / 255.0) * alpha
        mask_alpha_3d = mask_alpha[:, :, None]
        result = (result.astype(np.float32) * (1 - mask_alpha_3d) + 
                 colored_overlay.astype(np.float32) * mask_alpha_3d).astype(np.uint8)
    return result

def process_image_for_masks(image_path: str, interpreter, input_details, output_details, conf_threshold: float = 0.85):
    """Process a single image and return the nail mask and original image."""
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        print(f"Error: Could not read image {image_path}")
        return None

    H, W = frame_bgr.shape[:2]
    outputs, (in_w, in_h) = infer_tflite(interpreter, input_details, output_details, frame_bgr)
    det_idx, proto_idx, outputs_all = parse_tflite_outputs(outputs)

    combined_mask = np.zeros((H, W), dtype=np.uint8)

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

            valid = scores > conf_threshold
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

    # Refine and dilate mask
    if combined_mask.sum() > 0:
        refined_mask = refine_nail_mask(combined_mask, min_area=MIN_CONTOUR)
        kernel = np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
    else:
        refined_mask = combined_mask

    return refined_mask

def main():
    parser = argparse.ArgumentParser(description="Process images to generate nail masks only")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .tflite model")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to input images")
    parser.add_argument("--output_dir", default="nail_masks", help="Output directory for masks")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    interpreter, input_details, output_details = load_tflite_model(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(args.images)} images...")
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}. Skipping.")
            continue

        print(f"Processing: {os.path.basename(image_path)}")
        original_image = cv2.imread(image_path)
        mask = process_image_for_masks(image_path, interpreter, input_details, output_details, args.conf)
        
        if mask is not None and original_image is not None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save mask overlay on original image
            overlay_image = overlay_mask_on_image(original_image, mask, color=(199, 21, 133), alpha=0.5)
            overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_image)
            print(f"  Saved overlay to: {overlay_path}")
            
            # Also save the mask alone (optional)
            mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)
            print(f"  Saved mask to: {mask_path}")
        else:
            print(f"  Failed to process {image_path}")

    print(f"\nDone! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

