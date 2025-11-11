# Virtual Nail Polish Try-On Web Application

A web application that uses a TFLite model to apply virtual nail polish to images in real-time.

## Features

- 📸 Upload images and apply virtual nail polish
- 📷 Real-time webcam processing
- 🎨 Customizable nail polish colors
- ✨ Adjustable texture preservation
- 💎 Optional sheen effect

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8080
```

Note: If port 8080 is also in use, you can specify a different port by setting the PORT environment variable:
```bash
PORT=5001 python app.py
```

## Usage

1. **Upload an Image**: Click "Upload Image" and select an image file
2. **Start Webcam**: Click "Start Webcam" to use your camera for real-time processing
3. **Adjust Settings**:
   - Choose a nail polish color using the color picker
   - Adjust texture preservation (0 = full color, 1 = preserve original texture)
   - Enable/disable sheen effect
4. **Process**: Click "Process Image" or let webcam process automatically
5. **Clear**: Click "Clear" to reset and start over

## API Endpoints

### POST `/api/process`
Process an uploaded image file.

**Form Data:**
- `image`: Image file
- `color`: Hex color code (e.g., "#c71585")
- `texture`: Texture preservation value (0.0-1.0)
- `sheen`: "true" or "false"

### POST `/api/process_base64`
Process a base64-encoded image (for webcam).

**JSON Body:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "color": "#c71585",
  "texture": 0.35,
  "sheen": false
}
```

## Model

The application uses the TFLite model: `nails_seg_s_yolov8_v1_float16.tflite`

Make sure this file is in the same directory as `app.py`.

## Troubleshooting

- **Model not found**: Ensure `nails_seg_s_yolov8_v1_float16.tflite` is in the project directory
- **Webcam not working**: Check browser permissions for camera access
- **Processing errors**: Check that all dependencies are installed correctly

