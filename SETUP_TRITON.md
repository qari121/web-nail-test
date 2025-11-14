# NVIDIA Triton Setup for GPU Acceleration

## Quick Setup (5-10 minutes)

Your pod has 3x NVIDIA B200 GPUs but TensorFlow Lite GPU delegate isn't available. Triton is the best solution.

### Step 1: Check Docker
```bash
docker --version
# If not installed, install it:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Step 2: Create Triton Model Repository
```bash
cd /workspace/web-nail-test

# Create directory structure
mkdir -p triton_models/nail_seg/1

# Copy your model
cp nails_seg_s_yolov8_v1_float16.tflite triton_models/nail_seg/1/model.tflite
```

### Step 3: Create Triton Config
```bash
cat > triton_models/nail_seg/config.pbtxt << 'EOF'
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
EOF
```

### Step 4: Run Triton Server
```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Run Triton (use GPU 0)
docker run --gpus device=0 -d \
  --name triton-server \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --log-verbose=1

# Check if it's running
docker logs triton-server | tail -20
```

### Step 5: Test Triton
```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Check GPU usage
nvidia-smi
```

### Step 6: Update Flask App to Use Triton

I'll create a new version that uses Triton. For now, you can test Triton is working.

## Expected Performance

- **Current (CPU)**: 3 FPS, 54ms latency
- **With Triton GPU**: 15-30 FPS, 20-40ms latency (5-10x improvement)

## Alternative: Quick CPU Optimizations

If you want immediate improvements without Triton setup, I can optimize the CPU path further.

