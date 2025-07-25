import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "cyberpunkpor/wav2lip-onnx-HQ"
DEST_ROOT = os.getcwd()  # dynamically set to current dir

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch_and_place(repo_subpath):
    local_path = os.path.join(DEST_ROOT, repo_subpath)
    ensure_dir(os.path.dirname(local_path))
    print(f"⬇️ Downloading: {repo_subpath}")
    file_path = hf_hub_download(repo_id=REPO_ID, filename=repo_subpath, cache_dir="/tmp/hf_cache")

    # Resolve symlink if needed
    real_file_path = os.path.realpath(file_path)

    # Remove old file/symlink
    if os.path.exists(local_path) or os.path.islink(local_path):
        os.remove(local_path)

    # Copy actual model file
    shutil.copyfile(real_file_path, local_path)

    # Double-check copy result
    if os.path.islink(local_path):
        print(f"❌ Still a symlink: {local_path}")
    elif os.path.isfile(local_path):
        print(f"✅ Copied real file: {local_path}")
    else:
        print(f"❌ Something went wrong for {local_path}")

MODEL_FILES = [
    "blendmasker/blendmasker.onnx",
    "checkpoints/wav2lip.onnx",
    "checkpoints/wav2lip_gan.onnx",
    "enhancers/Codeformer/codeformer.onnx",
    "enhancers/GFPGAN/GFPGANv1.4.onnx",
    "enhancers/GPEN/GPEN-BFR-256.onnx",
    "enhancers/RealEsrgan/clear_reality_x4.onnx",
    "enhancers/restoreformer/restoreformer.onnx",
    "faceID/recognition.onnx",
    "resemble_denoiser/denoiser.onnx",
    "resemble_denoiser/denoiser_fp16.onnx",
    "xseg/xseg.onnx"
]

for f in MODEL_FILES:
    fetch_and_place(f)

print(f"✅ All model files downloaded/copied as real files in {DEST_ROOT}/")
