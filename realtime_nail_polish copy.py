import argparse
import os
import time
import threading
import queue
import traceback
import cv2
import numpy as np
import multiprocessing

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
DEFAULT_MODEL = "/Users/qari/Desktop/Farhan AI/nails_seg_s_yolov8_v1_float16.tflite"
TARGET_CAM = 0
DISPLAY_WINDOW = "Virtual Nail Polish. Press Q to quit."
DESIRED_FPS = 25            # UI/display FPS (what you see)
INFER_FPS = 8               # how often we actually run inference (lower = faster/less CPU)
TFLITE_THREADS = max(1, multiprocessing.cpu_count() - 1)
NAIL_COLOR = (199, 21, 133)  # fallback RGB
TEXTURE_DEFAULT = 0.35
MIN_CONTOUR = 60
DILATION_PIXELS = 4
FEATHER_IN = 3
FEATHER_OUT = 6
MAX_OUT_ALPHA = 0.08
MASK_DOWNSCALE = 0.5        # run heavy morphology/dist transforms on this scale (0.5 = half-size)
SHEEN_BASE_KERNEL = 71      # base kernel used in add_natural_sheen; scaled with mask size

# ---------------- helper functions (based on your code) ----------------
def adaptive_smooth_contour(pts, iterations=1):
    pts = np.array(pts, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new = []
        for i in range(len(pts)):
            p_prev = pts[(i-1) % len(pts)]
            p_curr = pts[i]
            p_next = pts[(i+1) % len(pts)]
            smoothed = 0.5 * p_curr + 0.25 * p_prev + 0.25 * p_next
            new.append(smoothed)
        pts = np.array(new)
    return pts.astype(np.int32)

def refine_nail_mask(bin_mask, min_area=60):
    # bin_mask expected uint8 0/255
    bin_mask = (bin_mask > 127).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return bin_mask
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k_small, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_small, iterations=1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(m)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) >= 3:
            smooth = adaptive_smooth_contour(approx[:,0,:], iterations=1)
            if len(smooth) >= 3:
                cv2.fillPoly(refined, [smooth], 255)
    refined = cv2.GaussianBlur(refined, (3,3), 0)
    _, refined = cv2.threshold(refined, 100, 255, cv2.THRESH_BINARY)
    return refined

def create_feathered_alpha(mask, inner_feather=3, outer_feather=6, max_out_alpha=0.10):
    m = (mask > 127).astype(np.uint8) * 255
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)
    # distance transforms are expensive; keep them but on small masks (worker scales)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5).astype(np.float32)
    h, w = m.shape
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[m > 0] = 1.0
    if outer_feather > 0:
        falloff = np.clip(1.0 - (dist_out / float(outer_feather)), 0.0, 1.0)
        alpha_out = falloff * float(max_out_alpha)
        alpha = np.where(m > 0, alpha, alpha_out)
    if inner_feather > 0:
        inside_fall = np.clip(dist_in / float(inner_feather), 0.0, 1.0)
        alpha_inside = 0.90 + 0.10 * inside_fall
        alpha = np.where(m > 0, alpha_inside, alpha)
    k = max(3, int(outer_feather) * 2 + 1)
    if k % 2 == 0:
        k += 1
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return alpha

def colorize_nail_texture(orig_rgb, alpha, color_rgb, texture_strength=0.35, brightness_adjust=1.0):
    img_f = orig_rgb.astype(np.float32)
    img_f = img_f * brightness_adjust
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    highlights = np.power(gray, 0.5)
    color = np.array(color_rgb, dtype=np.float32).reshape(1,1,3)
    colored_layer = img_f * texture_strength + color * (1.0 - texture_strength)
    highlight_boost = highlights[:,:,None] * 0.2
    colored_layer = colored_layer * (1 + highlight_boost)
    alpha_3 = alpha[:,:,None]
    out = img_f * (1.0 - alpha_3) + colored_layer * alpha_3
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def add_natural_sheen(img_rgb, hard_mask, intensity=0.06):
    gloss = np.zeros(hard_mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cx = x + int(w * 0.4)
        cy = y + int(h * 0.25)
        rx = max(1, int(w * 0.35))
        ry = max(1, int(h * 0.15))
        cv2.ellipse(gloss, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    # kernel size will be adjusted by worker proportional to mask size for speed
    g = (gloss.astype(np.float32) / 255.0) * intensity
    g3 = np.stack([g,g,g], axis=2)
    out = img_rgb.astype(np.float32) * (1 - g3) + 255.0 * g3
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# ---------------- TFLite helpers ----------------
def load_tflite_interpreter(path, num_threads=TFLITE_THREADS):
    if not os.path.exists(path):
        raise FileNotFoundError("TFLite model not found at: {}".format(path))
    # create interpreter with specified threads for better perf
    interpreter = tf.lite.Interpreter(model_path=path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def infer_tflite_once(interpreter, input_details, output_details, image_bgr):
    inp_d = input_details[0]
    shape = inp_d['shape']
    if len(shape) == 4:
        in_h, in_w = int(shape[1]), int(shape[2])
    else:
        in_h, in_w = 640, 640
    img = cv2.resize(image_bgr, (in_w, in_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if inp_d['dtype'] == np.uint8:
        input_tensor = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
    else:
        input_tensor = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    return outputs, (in_w, in_h)

def parse_tflite_outputs(outputs):
    det_idx = None
    proto_idx = None
    P = None
    for i,o in enumerate(outputs):
        if o.ndim == 4 and o.shape[-1] >= 1 and o.shape[-1] <= 1024:
            proto_idx = i
            P = o.shape[-1]
            break
        if o.ndim == 3 and o.shape[-1] >= 1 and o.shape[-1] <= 1024 and o.shape[0] <= 1024:
            proto_idx = i
            P = o.shape[-1]
            break
    if P is not None:
        for i,o in enumerate(outputs):
            if i == proto_idx:
                continue
            if o.ndim >= 2 and o.shape[-1] == 5 + P:
                det_idx = i
                break
    if proto_idx is None:
        for i,o in enumerate(outputs):
            if o.ndim in (3,4) and o.size > 1000:
                proto_idx = i
                P = o.shape[-1] if o.ndim in (3,4) else None
                break
    if det_idx is None:
        for i,o in enumerate(outputs):
            if i == proto_idx:
                continue
            if o.ndim >= 2 and o.shape[-1] >= 6:
                det_idx = i
                break
    return det_idx, proto_idx, outputs

# ---------------- Video reader thread. keeps the latest frame only ----------------
class CameraReader(threading.Thread):
    def __init__(self, src=0, width=640, height=480):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)  # macOS doesn't use CAP_DSHOW
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if not self.q.empty():
                try:
                    _ = self.q.get_nowait()
                except:
                    pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass
        self.cap.release()

    def read(self, timeout=0.01):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False

# ---------------- Inference worker ----------------
class InferenceWorker(threading.Thread):
    def __init__(self, model_path, input_details, output_details, num_threads=TFLITE_THREADS,
                 infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.input_details = input_details
        self.output_details = output_details
        self.infer_fps = float(infer_fps)
        self.mask_downscale = float(mask_downscale)
        self.in_q = queue.Queue(maxsize=1)       # latest raw frame (BGR)
        self.out_lock = threading.Lock()
        self.latest_result = None                # dict: {'rgb':..., 'ts':..., 'proc_dt':..., 'inf_dt':...}
        self.running = True
        # create interpreter here to avoid repeated creation
        self.interpreter, self.inp_d, self.out_d = load_tflite_interpreter(self.model_path, num_threads=num_threads)

    def submit(self, frame_bgr):
        # store latest frame; drop older
        if frame_bgr is None:
            return
        if not self.in_q.empty():
            try:
                _ = self.in_q.get_nowait()
            except:
                pass
        try:
            self.in_q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def run(self):
        last_inf = 0.0
        while self.running:
            try:
                # throttle inference rate
                now = time.time()
                if self.infer_fps > 0 and (now - last_inf) < (1.0 / self.infer_fps):
                    time.sleep(0.005)
                    continue
                try:
                    frame = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                ts0 = time.time()
                # run inference + build combined mask
                inf_start = time.time()
                outputs, (in_w, in_h) = infer_tflite_once(self.interpreter, self.inp_d, self.out_d, frame)
                det_idx, proto_idx, outputs_all = parse_tflite_outputs(outputs)
                inf_dt = time.time() - inf_start

                H, W = frame.shape[:2]

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
                        det = det.transpose(1,0)

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
                            boxes = det[:, :4] if det.shape[1] >= 4 else np.zeros((det.shape[0],4))

                        valid = scores > 0.25
                        boxes = boxes[valid]
                        mask_coeffs = mask_coeffs[valid]

                        ph, pw = proto.shape[0], proto.shape[1]
                        for i_det in range(len(mask_coeffs)):
                            coeff = mask_coeffs[i_det]
                            mask = np.tensordot(proto, coeff, axes=([2],[0]))
                            mask = 1.0 / (1.0 + np.exp(-mask))
                            mask_in = cv2.resize((mask * 255.0).astype(np.uint8), (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                            cx, cy, bw, bh = boxes[i_det]
                            if cx > 1.5 or cy > 1.5 or bw > 1.5 or bh > 1.5:
                                x1 = int(cx - bw/2)
                                y1 = int(cy - bh/2)
                                x2 = int(cx + bw/2)
                                y2 = int(cy + bh/2)
                            else:
                                x1 = int((cx - bw/2) * in_w)
                                y1 = int((cy - bh/2) * in_h)
                                x2 = int((cx + bw/2) * in_w)
                                y2 = int((cy + bh/2) * in_h)
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
                    # fallback scanning outputs for per-instance masks
                    found_masks = False
                    masks_tensor = None
                    for o in outputs_all:
                        try:
                            if proto_idx is not None and o is outputs_all[proto_idx]:
                                continue
                            if det_idx is not None and o is outputs_all[det_idx]:
                                continue
                        except:
                            pass
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

                # if no mask, produce quick empty result
                if combined_mask.sum() == 0:
                    final_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    proc_dt = time.time() - ts0
                else:
                    # downscale mask for heavy ops
                    small_w = max(8, int(W * self.mask_downscale))
                    small_h = max(8, int(H * self.mask_downscale))
                    small_mask = cv2.resize(combined_mask, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
                    refined_small = refine_nail_mask(small_mask, min_area=max(30, int(MIN_CONTOUR * (self.mask_downscale**2))))
                    # small dilation
                    k = max(1, int(DILATION_PIXELS * self.mask_downscale))
                    kernel = np.ones((k, k), np.uint8)
                    refined_small = cv2.dilate(refined_small, kernel, iterations=1)
                    # create alpha on small mask with feather sizes scaled
                    alpha_small = create_feathered_alpha(refined_small,
                                                         inner_feather=max(1, int(FEATHER_IN * self.mask_downscale)),
                                                         outer_feather=max(1, int(FEATHER_OUT * self.mask_downscale)),
                                                         max_out_alpha=MAX_OUT_ALPHA)
                    # upscale alpha to full size
                    alpha_map = cv2.resize(alpha_small, (W, H), interpolation=cv2.INTER_LINEAR)
                    orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    colored = colorize_nail_texture(orig_rgb, alpha_map, InferenceWorker.shared_color_rgb, texture_strength=InferenceWorker.shared_texture, brightness_adjust=1.05)
                    # optionally add sheen: compute gloss on scaled refined_small, then scale up and apply
                    if InferenceWorker.shared_sheen:
                        # scale sheen kernel proportional to small mask
                        scale = max(1.0, MASK_DOWNSCALE)
                        ksize = int(max(3, SHEEN_BASE_KERNEL * self.mask_downscale))
                        # add natural sheen on upscaled mask(image)
                        # but optimize: compute sheen on upscaled refined mask but with smaller blur kernel
                        refined_up = cv2.resize(refined_small, (W, H), interpolation=cv2.INTER_NEAREST)
                        # reduce heavy blur by using smaller kernel
                        final = add_natural_sheen(colored, refined_up, intensity=InferenceWorker.shared_sheen_intensity)
                    else:
                        final = colored
                    final_rgb = final
                    proc_dt = time.time() - ts0

                # publish result
                with self.out_lock:
                    self.latest_result = {
                        'rgb': final_rgb,
                        'ts': time.time(),
                        'inf_dt': inf_dt,
                        'proc_dt': proc_dt
                    }
                last_inf = time.time()
            except Exception as e:
                # catch everything in worker so it doesn't die
                print("Worker error:", e)
                traceback.print_exc()
                time.sleep(0.05)

    def get_latest(self):
        with self.out_lock:
            return None if self.latest_result is None else dict(self.latest_result)

    def stop(self):
        self.running = False

# static shared settings for worker behavior (updated once at startup)
InferenceWorker.shared_color_rgb = (199,21,133)
InferenceWorker.shared_texture = TEXTURE_DEFAULT
InferenceWorker.shared_sheen = False
InferenceWorker.shared_sheen_intensity = 0.06

# ---------------- Main processing ----------------
def run_live(model_path, cam_index=0, desired_fps=DESIRED_FPS, color_hex="#c71585",
             add_sheen=False, sheen_intensity=0.06, texture_amt=TEXTURE_DEFAULT,
             infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE):
    # parse color hex
    hexv = color_hex.lstrip('#')
    if len(hexv) == 6:
        try:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            color_rgb = (r, g, b)
        except:
            color_rgb = NAIL_COLOR
    else:
        color_rgb = NAIL_COLOR

    InferenceWorker.shared_color_rgb = color_rgb
    InferenceWorker.shared_texture = texture_amt
    InferenceWorker.shared_sheen = bool(add_sheen)
    InferenceWorker.shared_sheen_intensity = float(sheen_intensity)

    interpreter, input_details, output_details = load_tflite_interpreter(model_path, num_threads=TFLITE_THREADS)

    # start camera reader
    reader = CameraReader(src=cam_index, width=640, height=480)
    reader.start()

    # start worker (worker will create its own interpreter)
    worker = InferenceWorker(model_path, input_details, output_details,
                             num_threads=TFLITE_THREADS,
                             infer_fps=infer_fps,
                             mask_downscale=mask_downscale)
    worker.start()

    print("Model loaded. Starting camera. Press Q to quit.")
    last_time = time.time()
    fps_avg = 0.0
    last_display = None
    
    # Wait a moment for camera to initialize
    time.sleep(0.5)

    try:
        while True:
            start = time.time()
            ok, frame = reader.read(timeout=0.02)
            if not ok or frame is None:
                # Give camera more time on first few frames
                time.sleep(0.01)
                # still display last processed
                latest = worker.get_latest()
                if latest is not None:
                    display_rgb = latest['rgb']
                else:
                    # nothing yet, skip
                    time.sleep(0.005)
                    continue
            else:
                # submit newest frame to worker (non-blocking)
                worker.submit(frame)
                # show latest processed if available, otherwise raw
                latest = worker.get_latest()
                if latest is not None:
                    display_rgb = latest['rgb']
                else:
                    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # show fps (display fps)
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_curr = 1.0 / dt
                fps_avg = fps_avg * 0.85 + fps_curr * 0.15
            else:
                fps_curr = 0.0
            text = f"FPS {fps_avg:.1f}"

            # overlay small status: inference fps and processing time
            latest = worker.get_latest()
            if latest is not None:
                proc_ms = latest.get('proc_dt', 0.0) * 1000.0
                inf_ms = latest.get('inf_dt', 0.0) * 1000.0
                status = f"INF {inf_ms:.0f}ms PRC {proc_ms:.0f}ms"
            else:
                status = "INF - PRC -"

            # convert to bgr and display
            disp_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(disp_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(disp_bgr, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.imshow(DISPLAY_WINDOW, disp_bgr)

            # limit display FPS
            loop_time = time.time() - start
            target_frame_time = 1.0 / float(desired_fps)
            if loop_time < target_frame_time:
                time.sleep(max(0, target_frame_time - loop_time))

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        worker.stop()
        worker.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Stopped camera.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to .tflite model.")
    ap.add_argument("--cam", type=int, default=TARGET_CAM, help="Camera index.")
    ap.add_argument("--fps", type=int, default=DESIRED_FPS, help="Target display FPS.")
    ap.add_argument("--color", type=str, default="#c71585", help="Polish hex color.")
    ap.add_argument("--sheen", action="store_true", help="Enable sheen.")
    ap.add_argument("--sheen_int", type=float, default=0.06, help="Sheen intensity.")
    ap.add_argument("--texture", type=float, default=TEXTURE_DEFAULT, help="Texture preservation.")
    ap.add_argument("--infer_fps", type=float, default=INFER_FPS, help="Max inference FPS (lower => faster UI).")
    ap.add_argument("--mask_downscale", type=float, default=MASK_DOWNSCALE, help="Run heavy mask ops on downscaled mask (0.3-1.0).")
    args = ap.parse_args()
    run_live(args.model, cam_index=args.cam, desired_fps=args.fps, color_hex=args.color,
             add_sheen=args.sheen, sheen_intensity=args.sheen_int, texture_amt=args.texture,
             infer_fps=args.infer_fps, mask_downscale=args.mask_downscale)