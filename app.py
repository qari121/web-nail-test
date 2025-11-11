import os
import io
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nails_seg_s_yolov8_v1_float16.tflite")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global model interpreter (loaded once)
interpreter = None
input_details = None
output_details = None
det_idx = None
proto_idx = None

# Processing parameters
MIN_CONTOUR = 60
DILATION_PIXELS = 3

_SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def load_model():
    """Load TFLite model once at startup"""
    global interpreter, input_details, output_details, det_idx, proto_idx
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found at: {MODEL_PATH}")
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Parse outputs to find detection and proto indices
    dummy_outputs = []
    for od in output_details:
        shape = od['shape']
        dtype = od['dtype']
        dummy_outputs.append(np.zeros(shape, dtype=dtype))
    
    det_idx, proto_idx, _ = parse_tflite_outputs(dummy_outputs)
    print(f"Model loaded. Det idx: {det_idx}, Proto idx: {proto_idx}")


def parse_tflite_outputs(outputs):
    """Parse TFLite outputs to find detection and proto indices"""
    det_idx = None
    proto_idx = None
    P = None
    
    for i, o in enumerate(outputs):
        if o.ndim == 4 and o.shape[-1] >= 1 and o.shape[-1] <= 1024:
            proto_idx = i
            P = o.shape[-1]
            break
        if o.ndim == 3 and o.shape[-1] >= 1 and o.shape[-1] <= 1024 and o.shape[0] <= 1024:
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


def refine_nail_mask(bin_mask, min_area=60, kernel=_SMALL_KERNEL):
    """Optimized mask refinement"""
    bin_mask = (bin_mask > 127).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return bin_mask
    
    m = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(m)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        epsilon = 0.015 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) >= 3:
            cv2.fillPoly(refined, [approx], 255)
    
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    _, refined = cv2.threshold(refined, 100, 255, cv2.THRESH_BINARY)
    return refined


def overlay_mask(image_bgr, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay mask on image"""
    result = image_bgr.copy()
    if mask is not None and mask.sum() > 0:
        colored_overlay = np.zeros_like(result)
        colored_overlay[:, :] = color  # BGR color
        mask_alpha = (mask.astype(np.float32) / 255.0) * alpha
        mask_alpha_3d = mask_alpha[:, :, None]
        result = (result.astype(np.float32) * (1 - mask_alpha_3d) + 
                 colored_overlay.astype(np.float32) * mask_alpha_3d).astype(np.uint8)
    return result


def process_image(image_bgr):
    """Process image and return with nail masks overlaid"""
    global interpreter, input_details, output_details, det_idx, proto_idx
    
    H, W = image_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Get input shape
    input_shape = input_details[0]['shape']
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    
    # Resize and prepare input
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    
    if input_dtype == np.uint8:
        input_tensor = np.expand_dims(resized.astype(np.uint8), axis=0)
    else:
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    # Copy outputs immediately to avoid holding references to interpreter's internal data
    outputs = [np.copy(interpreter.get_tensor(od['index'])) for od in output_details]
    
    # Process outputs
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

    # Refine mask
    if combined_mask.any():
        refined_mask = refine_nail_mask(combined_mask, min_area=MIN_CONTOUR, kernel=_SMALL_KERNEL)
        dilate_kernel = np.ones((DILATION_PIXELS, DILATION_PIXELS), dtype=np.uint8)
        refined_mask = cv2.dilate(refined_mask, dilate_kernel, iterations=1)
    else:
        refined_mask = combined_mask

    # Overlay mask on original image (green color, 50% opacity)
    result_bgr = overlay_mask(image_bgr, refined_mask, color=(0, 255, 0), alpha=0.5)
    
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def image_to_base64(image_rgb):
    """Convert RGB image to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/process_base64', methods=['POST'])
def process_base64():
    """Process base64 encoded image (for webcam)"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process image
        result_rgb = process_image(image_bgr)
        
        # Convert to base64
        img_base64 = image_to_base64(result_rgb)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Loading TFLite model...")
    load_model()
    print("Model loaded successfully!")
    print("Starting Flask server...")
    port = int(os.environ.get('PORT', 8080))
    
    # Try HTTPS by default for webcam access
    try:
        print(f"Server running on https://0.0.0.0:{port}")
        print("Note: You may need to accept a security warning for the self-signed certificate")
        print("Access at: https://localhost:8080")
        app.run(debug=True, host='0.0.0.0', port=port, ssl_context='adhoc')
    except Exception as e:
        print(f"HTTPS failed: {e}")
        print(f"Falling back to HTTP at http://0.0.0.0:{port}")
        print("Note: Webcam may not work over HTTP. Try disabling browser extensions.")
        app.run(debug=True, host='0.0.0.0', port=port)
