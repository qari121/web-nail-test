# NVIDIA Triton Inference Server Setup Guide

## Quick Setup for RunPod H100

### 1. Install Docker (if not already installed)

```bash
# Check if Docker is installed
docker --version

# If not installed, install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. Create Model Repository Structure

```bash
# Create directory structure
mkdir -p triton_models/nail_seg/1

# Copy your TFLite model
cp nails_seg_s_yolov8_v1_float16.tflite triton_models/nail_seg/1/model.tflite
```

### 3. Create Triton Configuration

Create `triton_models/nail_seg/config.pbtxt`:

```protobuf
name: "nail_seg"
platform: "tensorflow_lite"
max_batch_size: 8

input [
  {
    name: "serving_default_input:0"
    data_type: TYPE_FP32
    dims: [ 640, 640, 3 ]
  }
]

output [
  {
    name: "StatefulPartitionedCall:0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "StatefulPartitionedCall:1"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 1, 2, 4 ]
}

optimization {
  execution_accelerators {
    gpu_execution_accelerator [
      {
        name: "tensorrt"
        parameters {
          key: "precision_mode"
          value: "FP16"
        }
      }
    ]
  }
}
```

**Note**: You may need to adjust input/output names based on your model. Check with:
```python
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print("Input:", interpreter.get_input_details())
print("Output:", interpreter.get_output_details())
```

### 4. Run Triton Server

```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Run Triton server
docker run --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

### 5. Test Triton Server

```bash
# Check server status
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Get model metadata
curl http://localhost:8000/v2/models/nail_seg
```

### 6. Update Flask App to Use Triton

Install Triton client:
```bash
pip install tritonclient[http]
```

Update `app.py` to use Triton:

```python
import tritonclient.http as httpclient
import numpy as np

# Initialize Triton client
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

def process_image_triton(image_bgr):
    """Process image using Triton Inference Server"""
    H, W = image_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # Normalize if needed
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Create inference request
    inputs = [httpclient.InferInput("serving_default_input:0", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    
    # Request outputs
    outputs = [
        httpclient.InferRequestedOutput("StatefulPartitionedCall:0"),
        httpclient.InferRequestedOutput("StatefulPartitionedCall:1")
    ]
    
    # Run inference
    response = triton_client.infer("nail_seg", inputs, outputs=outputs)
    
    # Get outputs
    proto_output = response.as_numpy("StatefulPartitionedCall:0")
    det_output = response.as_numpy("StatefulPartitionedCall:1")
    
    # Process outputs (same as before)
    # ... rest of processing logic ...
    
    return processed_image
```

### 7. Performance Comparison

**Before (TFLite CPU)**:
- Latency: 300-400ms
- FPS: 2-5

**After (Triton GPU)**:
- Latency: 50-100ms (expected)
- FPS: 15-30 (expected)

### 8. Troubleshooting

**Issue**: Model not loading
```bash
# Check Triton logs
docker logs <container_id>

# Verify model files
ls -la triton_models/nail_seg/1/
```

**Issue**: GPU not detected
```bash
# Check GPU
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Issue**: Input/output shape mismatch
- Check your model's actual input/output shapes
- Update `config.pbtxt` accordingly
- Use `interpreter.get_input_details()` and `interpreter.get_output_details()` to verify

### 9. Production Deployment

For production, use:
- Multiple Triton instances behind a load balancer
- Model versioning for A/B testing
- Monitoring with Prometheus/Grafana
- Auto-scaling based on request rate

```bash
# Run multiple instances
docker run --gpus all -p 8000:8000 ... tritonserver ... &
docker run --gpus all -p 8001:8000 ... tritonserver ... &
docker run --gpus all -p 8002:8000 ... tritonserver ... &
```

Then use Nginx to load balance:
```nginx
upstream triton_backend {
    least_conn;
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}
```

