import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nails_seg_s_yolov8_v1_float16.tflite")

# Global interpreter
interpreter = None
input_details = None
output_details = None
det_idx = None
proto_idx = None

# Processing parameters
MIN_CONTOUR = 60
DILATION_PIXELS = 3
MASK_DOWNSCALE = 0.25

_SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def refine_nail_mask(mask, min_area=60, kernel=None):
    """Refine mask by removing small contours."""
    if kernel is None:
        kernel = _SMALL_KERNEL
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.fillPoly(refined, [cnt], 255)
    return refined


def _resolve_mask_outputs(outputs, cached_det_idx, cached_proto_idx):
    """Resolve detection and proto indices from outputs."""
    if cached_det_idx is not None and cached_proto_idx is not None:
        return cached_det_idx, cached_proto_idx
    
    det_idx = None
    proto_idx = None
    
    for i, out in enumerate(outputs):
        shape = out.shape
        ndim = len(shape)
        
        # Detection output: usually 2D [N, features] or 3D [1, N, features]
        if ndim == 2 or (ndim == 3 and shape[0] == 1):
            if ndim == 3:
                shape = shape[1:]
            if shape[0] <= 100 and shape[1] >= 5:
                det_idx = i
        
        # Proto output: usually 3D [H, W, channels] or 4D [1, H, W, channels]
        if ndim == 3 or (ndim == 4 and shape[0] == 1):
            if ndim == 4:
                shape = shape[1:]
            if len(shape) == 3 and shape[2] >= 16:
                proto_idx = i
    
    return det_idx, proto_idx


def load_model():
    """Load TFLite model."""
    global interpreter, input_details, output_details, det_idx, proto_idx
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get dummy outputs to resolve indices
    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    dummy_outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    det_idx, proto_idx = _resolve_mask_outputs(dummy_outputs, None, None)
    
    print(f"Model loaded. Det idx: {det_idx}, Proto idx: {proto_idx}")


def process_image(image_bgr):
    """Process image and return mask overlay."""
    global interpreter, input_details, output_details, det_idx, proto_idx
    
    H, W = image_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    input_shape = input_details[0]['shape']
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    
    if input_dtype == np.uint8:
        input_tensor = np.expand_dims(resized.astype(np.uint8), axis=0)
    else:
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    outputs = [np.copy(interpreter.get_tensor(od['index'])) for od in output_details]
    
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    
    if proto_idx is not None and det_idx is not None:
        proto = outputs[proto_idx].copy()
        if proto.ndim == 4 and proto.shape[0] == 1:
            proto = proto[0].copy()
        elif proto.ndim == 3 and proto.shape[0] == 1:
            proto = proto[0].copy()
        if proto is not None and proto.ndim != 3:
            proto = None
        
        det = outputs[det_idx].copy()
        if det.ndim == 3 and det.shape[0] == 1:
            det = det[0].copy()
        if det.ndim == 2 and det.shape[0] < det.shape[1] and det.shape[0] <= 50:
            det = det.transpose(1, 0).copy()
        
        if proto is not None and det.ndim == 2:
            P = proto.shape[-1]
            cols = det.shape[1]
            if cols >= 5 + P:
                mask_coeffs = det[:, -P:].copy()
                scores = det[:, 4].copy()
                boxes = det[:, :4].copy()
            else:
                mask_coeffs = det[:, -P:].copy()
                scores = det[:, 4].copy() if det.shape[1] > 4 else np.zeros(det.shape[0])
                boxes = det[:, :4].copy() if det.shape[1] >= 4 else np.zeros((det.shape[0], 4))
            
            valid_idx = np.where(scores > 0.25)[0]
            if valid_idx.size > 0:
                mask_coeffs = mask_coeffs[valid_idx].copy()
                boxes = boxes[valid_idx].copy()
                
                ph, pw, _ = proto.shape
                proto_flat = proto.reshape(-1, P).copy()
                mask_logits = proto_flat @ mask_coeffs.T
                mask_stack = 1.0 / (1.0 + np.exp(-mask_logits))
                mask_stack = mask_stack.reshape(ph, pw, -1).copy()
                
                combined_mask_in = np.zeros((in_h, in_w), dtype=np.uint8)
                for idx in range(mask_stack.shape[-1]):
                    mask = mask_stack[:, :, idx].copy()
                    mask_in = cv2.resize((mask * 255.0).astype(np.uint8),
                                         (in_w, in_h),
                                         interpolation=cv2.INTER_AREA)
                    
                    cx, cy, bw, bh = boxes[idx]
                    if cx > 1.5 or cy > 1.5 or bw > 1.5 or bh > 1.5:
                        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                    else:
                        x1 = int((cx - bw / 2) * in_w)
                        y1 = int((cy - bh / 2) * in_h)
                        x2 = int((cx + bw / 2) * in_w)
                        y2 = int((cy + bh / 2) * in_h)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(in_w, x2), min(in_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        roi = mask_in[y1:y2, x1:x2]
                        if roi.size:
                            combined_mask_in[y1:y2, x1:x2] = np.maximum(
                                combined_mask_in[y1:y2, x1:x2], roi
                            )
                    else:
                        combined_mask_in = np.maximum(combined_mask_in, mask_in)
                
                if combined_mask_in.any():
                    mask_full = cv2.resize(combined_mask_in, (W, H), interpolation=cv2.INTER_AREA)
                    _, combined_mask = cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY)
    
    if not combined_mask.any():
        return frame_rgb
    
    # Process mask
    small_w = max(8, int(W * MASK_DOWNSCALE))
    small_h = max(8, int(H * MASK_DOWNSCALE))
    small_mask = cv2.resize(combined_mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
    _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
    
    refined_small = refine_nail_mask(small_mask, min_area=MIN_CONTOUR, kernel=_SMALL_KERNEL)
    dilate_kernel = np.ones((max(1, int(DILATION_PIXELS * MASK_DOWNSCALE)),
                            max(1, int(DILATION_PIXELS * MASK_DOWNSCALE))), dtype=np.uint8)
    refined_small = cv2.dilate(refined_small, dilate_kernel, iterations=1)
    
    # Overlay green mask
    green_color = (0, 255, 0)  # BGR for OpenCV
    alpha_map = cv2.resize(refined_small.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    alpha_3 = alpha_map[:, :, None]
    
    # Create a green overlay
    green_overlay = np.zeros_like(frame_rgb, dtype=np.float32)
    green_overlay[:, :, 1] = green_color[1]  # Green channel
    
    # Blend the green overlay with the original frame using the alpha map
    final_rgb = (frame_rgb.astype(np.float32) * (1.0 - alpha_3) + green_overlay * alpha_3).astype(np.uint8)
    
    return final_rgb


def base64_to_image(b64):
    """Convert base64 string to OpenCV image."""
    # Remove data URL prefix if present
    if ',' in b64:
        b64 = b64.split(',')[1]
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(img):
    """Convert OpenCV image to base64 string."""
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')


@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "Nail segmentation API"})


@app.post("/process")
def process():
    """Process image and return result with nail masks."""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        img_b64 = data["image"]
        img = base64_to_image(img_b64)
        
        if img is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400
        
        processed_img = process_image(img)
        result_b64 = image_to_base64(processed_img)
        
        return jsonify({
            "success": True,
            "image": result_b64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("Loading TFLite model...")
    load_model()
    print("Model loaded successfully!")
    
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
