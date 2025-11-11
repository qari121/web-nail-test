# Deploying to RunPod

## Step 1: Connect to Your Pod

You can use either:
- **SSH:** `ssh 71wvnkbiigepcg-64411b67@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **Web Terminal:** Enable it in the RunPod console

## Step 2: Clone Your Repository

```bash
git clone https://github.com/qari121/web-nail-test.git
cd web-nail-test
```

## Step 3: Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 4: Run the Flask App

```bash
python app.py
```

The app will start on port 8080 (or whatever PORT env var is set).

## Step 5: Expose the Port in RunPod

1. Go to your pod's "Connect" tab in RunPod console
2. Scroll to "HTTP Services" section
3. Click "+ Expose HTTP Service"
4. Set:
   - **Port:** 8080 (or whatever port your app uses)
   - **Name:** Flask App (or any name)
5. RunPod will provide you with a public URL like: `https://your-pod-id-8080.proxy.runpod.net`

## Step 6: Access Your App

Visit the URL provided by RunPod to access your web application!

## Notes:

- The app will stop when you disconnect SSH unless you run it in the background or use a process manager
- To keep it running: `nohup python app.py &` or use `screen`/`tmux`
- Make sure your TFLite model file (`nails_seg_s_yolov8_v1_float16.tflite`) is in the repository or upload it to the pod

