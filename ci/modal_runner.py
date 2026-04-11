import modal

app = modal.App("esm-efficient-tests")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
        "torch>=2.7.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "MAX_JOBS=4 CXX=g++ CC=gcc pip install flash-attn --no-build-isolation",
        gpu="L4",
    )
    .pip_install(
        "einops",
        "accelerate",
        "pandas",
        "numpy",
        "polars",
        "torchmetrics",
        "lightning",
        "scikit-learn",
        "huggingface_hub",
        "safetensors",
        "pytest",
        "pytest-runner",
        "pooch",
        "esm",
        "httpx",
        "bitsandbytes",
    )
    .run_commands(
        "pip install git+https://github.com/MuhammedHasan/fair-esm.git",
        # Patch fair-esm for PyTorch 2.6+ compatibility (weights_only default changed)
        "sed -i 's/torch.load(str(model_location), map_location=\"cpu\")/torch.load(str(model_location), map_location=\"cpu\", weights_only=False)/' /usr/local/lib/python3.11/site-packages/fair_esm/pretrained.py",
    )
)

# Cache downloaded model weights across runs (~800MB total)
model_cache = modal.Volume.from_name("esm-model-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    image=image,
    volumes={"/model-cache": model_cache},
    timeout=3600,
)
def run_tests(repo_url: str, ref: str):
    import os
    import shutil
    import subprocess

    # Clone the repo at the specific commit
    subprocess.run(["git", "clone", repo_url, "/app"], check=True)
    subprocess.run(["git", "checkout", ref], cwd="/app", check=True)

    # Restore cached model weights to avoid re-downloading each run
    cache_dir = "/model-cache/test-data"
    os.makedirs(cache_dir, exist_ok=True)
    for fname in os.listdir(cache_dir):
        dst = f"/app/tests/data/{fname}"
        if not os.path.exists(dst):
            shutil.copy2(f"{cache_dir}/{fname}", dst)

    # Install the package itself
    subprocess.run(["pip", "install", "-e", "."], cwd="/app", check=True)

    # Run tests
    result = subprocess.run(["pytest", "tests/", "-v", "--tb=short"], cwd="/app")

    # Save newly downloaded models to cache
    for fname in os.listdir("/app/tests/data"):
        if fname.endswith((".pt", ".pth")):
            dst = f"{cache_dir}/{fname}"
            if not os.path.exists(dst):
                shutil.copy2(f"/app/tests/data/{fname}", dst)
    model_cache.commit()

    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    repo_url: str = "https://github.com/hmtcelik/esm-efficient.git",
    ref: str = "master",
):
    run_tests.remote(repo_url, ref)
