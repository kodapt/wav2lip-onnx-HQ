#!/bin/bash
set -e

# ---------- PREPARE SYSTEM ----------
echo "Updating and installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg libsndfile1 git wget

# ---------- PYTHON PACKAGES ----------
echo "Upgrading pip and installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    onnxruntime-gpu==1.16.3 \
    huggingface-hub \
    opencv-python-headless \
    tqdm \
    numpy==1.24.4 \
    librosa==0.10.1 \
    numba==0.59.1 \
    imutils \
    soundfile \
    ffmpeg-python \
    pillow \
    scikit-image \
    scipy

# ---------- CLONE REPO ----------
echo "Cloning wav2lip-onnx-HQ repo..."
cd /workspace
if [ ! -d "wav2lip-onnx-HQ" ]; then
    git clone https://github.com/kodapt/wav2lip-onnx-HQ.git
fi

# ---------- DOWNLOAD MODELS ----------
echo "Running download_models.py (need HF_TOKEN set in env)..."
# Copy download_models.py to repo (adjust path if needed)
cp /workspace/ai-images/wav2liphq/download_models.py /workspace/wav2lip-onnx-HQ/download_models.py

# HuggingFace token must be set as environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HuggingFace token not set. Set HF_TOKEN env variable if download needs authentication."
fi

cd /workspace/wav2lip-onnx-HQ
python3 download_models.py

# ---------- PREP INPUT/OUTPUT DIRS ----------
echo "Creating input/output directories..."
mkdir -p /workspace/input /workspace/output

echo "Setup complete!"
echo "To run inference:"
echo "python3 inference_onnxModel.py \\"
echo "  --checkpoint_path /workspace/wav2lip-onnx-HQ/checkpoints/wav2lip_gan.onnx \\"
echo "  --face /workspace/input/video.mp4 \\"
echo "  --audio /workspace/input/sound.wav \\"
echo "  --outfile /workspace/output/output_video.mp4 \\"
echo "  --resize_factor 1 --fps 25.0 --blending 7 --pads 4 --face_mode 0 --hq_output --fade --enhancer gfpgan --sharpen --face_mask --headless"
