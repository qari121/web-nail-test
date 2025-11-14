from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# -------------------------
# MODEL LOADING
# -------------------------
MODEL_PATH = "nails_seg_s_yolov8_v1_float16.tflite"
model = YOLO(MODEL_PATH)


# -------------------------
# IMAGE HELPERS
# -------------------------
def base64_to_image(b64):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img):
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')


# -------------------------
# MASK PROCESSING
# -------------------------
def smooth_mask(mask):
    mask = cv2.GaussianBlur(mask, (15,15), 0)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), 1)
    mask = cv2.GaussianBlur(mask, (15,15), 0)
    return mask


# -------------------------
# APPLY GLOSSY NAIL POLISH
# -------------------------
def apply_glossy_pink(image, mask):
    pink = np.array([255, 77, 136], dtype=np.uint8)  # Solid Pink #ff4d88

    mask_f = (mask / 255.0).reshape(mask.shape[0], mask.shape[1], 1)

    # Base pink layer
    colored = image * (1 - mask_f) + pink * mask_f

    # Gloss highlight
    h, w = mask.shape
    highlight = np.linspace(0, 1, w)
    highlight = np.tile(highlight, (h, 1))
    highlight = cv2.GaussianBlur(highlight, (51,51), 0)
    highlight = highlight[..., None]

    glossy_intensity = 0.35
    glossy = colored + glossy_intensity * highlight * 255 * mask_f
    glossy = np.clip(glossy, 0, 255).astype(np.uint8)

    return glossy


# -------------------------
# APPLY POLISH TO ALL NAILS
# -------------------------
def render_nail_polish(img, masks):
    output = img.copy()

    for m in masks:
        m = smooth_mask(m)
        output = apply_glossy_pink(output, m)

    return output


# -------------------------
# MAIN API ROUTE
# -------------------------
@app.post("/process")
def process():
    data = request.json
    img_b64 = data["image"]

    # Decode image
    img = base64_to_image(img_b64)

    # Run inference
    result = model(img, imgsz=640)[0]

    # Collect individual nail masks
    masks = []
    if result.masks is not None:
        for i in range(result.masks.data.shape[0]):
            m = result.masks.data[i].cpu().numpy()
            m = (m * 255).astype(np.uint8)
            masks.append(m)

    # Apply nail polish effect
    final_img = render_nail_polish(img, masks)

    # Encode and return
    return jsonify({
        "image": image_to_base64(final_img)
    })


# -------------------------
# START SERVER
# -------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)