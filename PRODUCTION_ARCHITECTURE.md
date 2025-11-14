# Production Architecture Guide: Replicating Perfect Corp's Setup

## How Perfect Corp & Similar Services Deploy Their Models

### 1. **Architecture Overview**

Perfect Corp and similar real-time AI services use a **multi-tier architecture**:

```
Client (Browser/Mobile)
    ↓ WebRTC / WebSocket
CDN / Edge Servers (Geographic Distribution)
    ↓ Load Balancer
GPU Inference Servers (NVIDIA Triton / TensorFlow Serving)
    ↓ Model Cache
Optimized Models (TensorRT / ONNX Runtime / TFLite GPU)
```

### 2. **Key Components**

#### **A. Edge Computing & CDN**
- **Purpose**: Reduce latency by processing requests closer to users
- **Services**: Cloudflare Workers, AWS CloudFront, Fastly
- **Benefit**: 50-200ms latency reduction

#### **B. Model Serving Infrastructure**
- **NVIDIA Triton Inference Server** (Most Common)
  - Supports multiple frameworks (TensorFlow, PyTorch, ONNX, TFLite)
  - Automatic batching
  - Dynamic batching for throughput
  - GPU acceleration
  - Model versioning

- **TensorFlow Serving**
  - Optimized for TensorFlow models
  - REST/gRPC APIs
  - Model versioning

- **ONNX Runtime**
  - Cross-platform optimization
  - GPU acceleration via CUDA/ROCm
  - Quantization support

#### **C. GPU Acceleration**
- **TensorRT** (NVIDIA): Optimized inference engine
- **CUDA**: Direct GPU access
- **OpenVINO** (Intel): CPU/GPU optimization
- **CoreML** (Apple): Apple Silicon optimization

#### **D. Communication Protocols**
- **WebRTC**: Real-time video streaming (low latency)
- **WebSocket**: Bidirectional communication
- **gRPC**: Efficient RPC for model serving
- **HTTP/2**: Multiplexed requests

#### **E. Client-Side Optimizations**
- **WebAssembly (WASM)**: Run models in browser
- **WebGL/WebGPU**: GPU acceleration in browser
- **Progressive Enhancement**: Start with server, move to edge/client

---

## Current Setup vs. Production Setup

### **Current Setup (Your App)**
```
Browser → Base64 Image → Flask → TFLite (CPU) → Base64 Response
```
**Issues:**
- ❌ Base64 encoding/decoding overhead (~30-50ms)
- ❌ CPU-only inference (slow)
- ❌ Single server instance
- ❌ No edge caching
- ❌ Synchronous processing

### **Production Setup (Perfect Corp)**
```
Browser → WebRTC → Edge Server → GPU Inference → Optimized Response
```
**Benefits:**
- ✅ Direct video streaming (no encoding)
- ✅ GPU acceleration (10-50x faster)
- ✅ Geographic distribution (lower latency)
- ✅ Model caching at edge
- ✅ Asynchronous processing

---

## Step-by-Step: Replicating Perfect Corp's Architecture

### **Phase 1: Optimize Model Serving (Immediate Impact)**

#### **Option A: NVIDIA Triton Inference Server** (Recommended for H100)

1. **Install Triton**:
```bash
# On RunPod (Ubuntu)
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Create model repository structure
mkdir -p triton_models/nail_seg/1
cp nails_seg_s_yolov8_v1_float16.tflite triton_models/nail_seg/1/model.tflite

# Create config.pbtxt
cat > triton_models/nail_seg/config.pbtxt << EOF
name: "nail_seg"
platform: "tensorflow_lite"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 640, 640, 3 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
EOF
```

2. **Run Triton Server**:
```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

3. **Update Flask App to Use Triton**:
```python
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8000")

def process_image_triton(image_bgr):
    # Preprocess
    input_data = preprocess_image(image_bgr)
    
    # Create inference request
    inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    
    outputs = [httpclient.InferRequestedOutput("output")]
    
    # Run inference
    response = triton_client.infer("nail_seg", inputs, outputs=outputs)
    result = response.as_numpy("output")
    
    return postprocess_mask(result, image_bgr)
```

**Expected Improvement**: 5-10x faster inference (GPU acceleration)

---

#### **Option B: TensorFlow Serving** (Alternative)

1. **Install TensorFlow Serving**:
```bash
docker pull tensorflow/serving:latest-gpu

# Convert TFLite to SavedModel (if needed)
# Or use TFLite directly with custom serving
```

2. **Run TensorFlow Serving**:
```bash
docker run -p 8501:8501 --gpus all \
  -v $(pwd)/models:/models \
  tensorflow/serving:latest-gpu \
  --model_base_path=/models/nail_seg \
  --rest_api_port=8501
```

---

### **Phase 2: Optimize Communication (Reduce Latency)**

#### **Replace Base64 with WebRTC or WebSocket**

**Current (Base64)**:
- Encode: ~10-20ms
- Transfer: ~50-100ms (depends on size)
- Decode: ~10-20ms
- **Total overhead**: ~70-140ms

**WebRTC (Recommended)**:
```javascript
// Frontend: Capture video stream
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const peerConnection = new RTCPeerConnection();

// Send video frames directly (no encoding)
stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));
```

**WebSocket (Simpler Alternative)**:
```javascript
// Frontend: Send raw ImageData
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0);
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

// Send via WebSocket (binary)
ws.send(imageData.data.buffer);
```

**Expected Improvement**: 50-100ms latency reduction

---

### **Phase 3: GPU Acceleration (Maximum Performance)**

#### **Convert TFLite to TensorRT** (Best Performance)

1. **Install TensorRT**:
```bash
# On RunPod with CUDA
pip install nvidia-tensorrt
```

2. **Convert Model**:
```python
import tensorrt as trt

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Convert to ONNX first, then TensorRT
# (Requires ONNX conversion tool)
```

3. **Use TensorRT Runtime**:
```python
import tensorrt as trt
import pycuda.driver as cuda

# Load TensorRT engine
with open("model.trt", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
    
# Create execution context
context = engine.create_execution_context()

# Run inference on GPU
# (Much faster than TFLite CPU)
```

**Expected Improvement**: 10-50x faster than CPU TFLite

---

### **Phase 4: Edge Deployment (Geographic Distribution)**

#### **Option A: Cloudflare Workers** (Serverless Edge)

1. **Deploy Model to Cloudflare**:
```javascript
// workers/nail-seg.js
export default {
  async fetch(request) {
    const image = await request.arrayBuffer();
    
    // Run inference at edge (closest to user)
    const result = await runInference(image);
    
    return new Response(JSON.stringify(result));
  }
}
```

2. **Use Cloudflare's GPU Workers** (Beta):
- Deploy model to edge locations
- Automatic geographic routing
- Low latency (<50ms for most users)

#### **Option B: AWS Lambda@Edge**
- Similar to Cloudflare Workers
- Deploy to AWS edge locations

**Expected Improvement**: 50-200ms latency reduction (depending on user location)

---

### **Phase 5: Load Balancing & Scaling**

#### **Use Nginx or HAProxy**

```nginx
# nginx.conf
upstream inference_servers {
    least_conn;
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    
    location /api/inference {
        proxy_pass http://inference_servers;
        proxy_buffering off;
        proxy_request_buffering off;
    }
}
```

#### **Horizontal Scaling**
- Run multiple Triton/TensorFlow Serving instances
- Use Kubernetes for auto-scaling
- Load balance across instances

**Expected Improvement**: Handle 10-100x more concurrent users

---

## Quick Wins (Implement First)

### **1. Enable GPU Delegate Properly**

Your current code tries GPU delegate but may fail. Fix it:

```python
# Check if CUDA is available
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True)
if result.returncode == 0:
    # CUDA is available, use GPU delegate
    try:
        # For TFLite GPU delegate, need to build it first
        delegate = tf.lite.experimental.load_delegate('libgpu_delegate.so')
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[delegate]
        )
    except:
        # Fallback to CPU
        pass
```

### **2. Use WebSocket Instead of Base64**

Replace base64 transfer with WebSocket binary:

```python
# app.py
from flask_socketio import SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('frame')
def handle_frame(data):
    # data is binary image data (no base64 encoding)
    image = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    result = process_image(image_bgr)
    # Send back as binary
    _, buffer = cv2.imencode('.jpg', result)
    socketio.emit('result', buffer.tobytes())
```

**Expected Improvement**: 50-70ms latency reduction

### **3. Implement Request Batching**

Process multiple frames together:

```python
from queue import Queue
import threading

frame_queue = Queue(maxsize=10)

def batch_processor():
    batch = []
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
            batch.append(frame)
            if len(batch) >= 4:  # Process 4 frames at once
                results = process_batch(batch)
                # Send results
                batch = []
        except:
            if batch:
                results = process_batch(batch)
                batch = []
```

**Expected Improvement**: 2-3x throughput increase

---

## Recommended Architecture for Your Use Case

### **For RunPod H100 NVL 2x:**

```
┌─────────────┐
│   Browser   │
│  (WebRTC)   │
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│  Flask App      │
│  (WebSocket)    │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  NVIDIA Triton  │
│  (GPU Inference)│
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  TensorRT Model │
│  (Optimized)     │
└─────────────────┘
```

### **Implementation Priority:**

1. **Week 1**: Replace Base64 with WebSocket (50ms improvement)
2. **Week 2**: Deploy Triton Inference Server (5-10x improvement)
3. **Week 3**: Convert to TensorRT (additional 2-5x improvement)
4. **Week 4**: Add edge deployment (50-200ms improvement)

---

## Performance Targets

### **Current Performance**:
- FPS: 2-5
- Latency: 300-400ms

### **After Optimizations**:
- **Phase 1 (WebSocket)**: FPS: 5-10, Latency: 200-300ms
- **Phase 2 (Triton GPU)**: FPS: 15-30, Latency: 50-100ms
- **Phase 3 (TensorRT)**: FPS: 30-60, Latency: 20-50ms
- **Phase 4 (Edge)**: FPS: 30-60, Latency: 10-30ms

---

## Cost Considerations

### **Current Setup**:
- RunPod H100: ~$4-8/hour
- Single instance

### **Production Setup**:
- RunPod H100: ~$4-8/hour (inference server)
- Cloudflare Workers: ~$5/month (edge)
- **Total**: Similar cost, 10-50x better performance

---

## Next Steps

1. **Immediate**: Implement WebSocket (see code above)
2. **Short-term**: Deploy Triton Inference Server
3. **Long-term**: Convert to TensorRT + Edge deployment

Would you like me to implement any of these optimizations?

