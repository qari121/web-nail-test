import os
import io
import base64
import traceback
import multiprocessing
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

# Optimize OpenCV
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Limit OpenCV threads to avoid contention

# Global model interpreter (loaded once)
interpreter = None
input_details = None
output_details = None
det_idx = None
proto_idx = None

# Processing parameters (optimized for performance - aggressive settings)
MIN_CONTOUR = 40  # Reduced for faster processing
DILATION_PIXELS = 2  # Reduced for faster processing
MASK_DOWNSCALE = 0.25  # Further reduced for maximum speed
TFLITE_THREADS = max(1, multiprocessing.cpu_count() - 1)  # Use available CPUs

_SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Pre-allocated buffers (will be initialized after model load)
_input_tensor = None
_float_buffer = None


def load_model():
    """Load TFLite model once at startup with optimizations"""
    global interpreter, input_details, output_details, det_idx, proto_idx
    global _input_tensor, _float_buffer
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found at: {MODEL_PATH}")
    
    # Use more threads for better performance on multi-core systems
    # Try GPU delegate if available (for CUDA-enabled RunPod instances)
    interpreter = None
    try:
        # Try GPU delegate first (if CUDA is available)
        delegate_list = []
        try:
            gpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
            delegate_list.append(gpu_delegate)
            print("Using EdgeTPU delegate")
        except:
            try:
                gpu_delegate = tf.lite.experimental.load_delegate('libgpu_delegate.so')
                delegate_list.append(gpu_delegate)
                print("Using GPU delegate")
            except:
                print("GPU delegate not available, using CPU")
        
        if delegate_list:
            interpreter = tf.lite.Interpreter(
                model_path=MODEL_PATH,
                experimental_delegates=delegate_list,
                num_threads=TFLITE_THREADS
            )
        else:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=TFLITE_THREADS)
    except Exception as e:
        print(f"GPU delegate failed ({e}), falling back to CPU")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=TFLITE_THREADS)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Pre-allocate input tensor and buffer for faster processing
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    
    if input_dtype == np.uint8:
        _input_tensor = np.empty(input_shape, dtype=np.uint8)
        _float_buffer = None
    else:
        _input_tensor = np.empty(input_shape, dtype=np.float32)
        _float_buffer = np.empty((in_h, in_w, input_shape[3]), dtype=np.float32)
    
    # Parse outputs to find detection and proto indices
    dummy_outputs = []
    for od in output_details:
        shape = od['shape']
        dtype = od['dtype']
        dummy_outputs.append(np.zeros(shape, dtype=dtype))
    
    det_idx, proto_idx, _ = parse_tflite_outputs(dummy_outputs)
    print(f"Model loaded. Det idx: {det_idx}, Proto idx: {proto_idx}")
    print(f"Using {TFLITE_THREADS} threads for inference")
    print(f"Mask downscale: {MASK_DOWNSCALE}")


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


def adaptive_smooth_contour(pts, iterations=1):
    """Optimized contour smoothing"""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)
    
    # Single iteration is usually enough
    for _ in range(min(iterations, 1)):
        if len(pts) < 3:
            break
        # Vectorized smoothing
        p_prev = np.roll(pts, 1, axis=0)
        p_next = np.roll(pts, -1, axis=0)
        pts = 0.5 * pts + 0.25 * p_prev + 0.25 * p_next
    
    return pts.astype(np.int32)


def refine_nail_mask(bin_mask, min_area=60, kernel=_SMALL_KERNEL):
    """Optimized mask refinement with contour smoothing"""
    bin_mask = (bin_mask > 127).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return bin_mask
    
    # Combine morphological operations
    m = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(m)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        # Reduced epsilon for faster approximation
        epsilon = 0.015 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) >= 3:
            # Smooth contours for better quality
            smooth = adaptive_smooth_contour(approx[:, 0, :], iterations=1)
            if len(smooth) >= 3:
                cv2.fillPoly(refined, [smooth], 255)
    
    # Lighter blur for performance
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
    """Process image and return with nail masks overlaid - Optimized for low latency"""
    global interpreter, input_details, output_details, det_idx, proto_idx
    global _input_tensor, _float_buffer
    
    H, W = image_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Get input shape
    input_shape = input_details[0]['shape']
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    
    # Use faster resize algorithm (LINEAR is faster than AREA for downscaling)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    
    # Use pre-allocated buffers for better performance
    if input_dtype == np.uint8:
        np.copyto(_input_tensor[0], resized, casting='unsafe')
    else:
        _float_buffer[...] = resized
        _float_buffer *= (1.0 / 255.0)
        np.copyto(_input_tensor[0], _float_buffer)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], _input_tensor)
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
                                         interpolation=cv2.INTER_LINEAR)

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
                    mask_full = cv2.resize(combined_mask_in, (W, H), interpolation=cv2.INTER_LINEAR)
                    _, combined_mask = cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY)

    # Refine mask with downscaling for better performance
    if combined_mask.any():
        # Downscale mask for faster processing
        small_w = max(8, int(W * MASK_DOWNSCALE))
        small_h = max(8, int(H * MASK_DOWNSCALE))
        small_mask = cv2.resize(combined_mask, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Refine on smaller mask
        min_area_scaled = max(20, int(MIN_CONTOUR * (MASK_DOWNSCALE ** 2)))  # Reduced threshold
        refined_small = refine_nail_mask(small_mask, min_area=min_area_scaled, kernel=_SMALL_KERNEL)
        
        # Dilate on smaller mask
        dilate_kernel_scaled = np.ones((
            max(1, int(DILATION_PIXELS * MASK_DOWNSCALE)),
            max(1, int(DILATION_PIXELS * MASK_DOWNSCALE))
        ), dtype=np.uint8)
        refined_small = cv2.dilate(refined_small, dilate_kernel_scaled, iterations=1)
        
        # Upscale back to original size (LINEAR is faster)
        refined_mask = cv2.resize(refined_small, (W, H), interpolation=cv2.INTER_LINEAR)
        _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
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
        
        # Limit image size for performance (aggressive reduction for speed)
        max_dimension = 480  # Further reduced for maximum performance
        if image_bgr is not None:
            h, w = image_bgr.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
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


# Load model at application startup (works with both Flask dev server and gunicorn)
# This runs when the module is imported, ensuring model is loaded before any requests
if interpreter is None:
    print("Loading TFLite model...")
    load_model()
    print("Model loaded successfully!")

if __name__ == '__main__':
    # Model already loaded above, but ensure it's loaded
    if interpreter is None:
        print("Loading TFLite model...")
        load_model()
        print("Model loaded successfully!")
    
    print("Starting Flask server...")
    
    # Get port from environment variable (RunPod/Render provides this)
    port = int(os.environ.get('PORT', 8080))
    
    # Use threaded mode for better concurrency
    # Bind to 0.0.0.0 to accept connections from all interfaces
    print(f"Server running on http://0.0.0.0:{port}")
    print(f"OpenCV threads: {cv2.getNumThreads()}")
    print(f"TensorFlow threads: {TFLITE_THREADS}")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
