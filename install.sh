conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

pip install asteroid-filterbanks==0.4.0
pip install librosa==0.9.2
pip install git+https://github.com/openai/whisper.git@7858aa9c08d98f75575035ecd6481f462d66ca27
pip install pandas==2.0.2
# Additional deps for model auto-download
pip install huggingface-hub requests

# Attempt to ensure the model is present before the app runs. This script will
# download from Hugging Face (HF_REPO_ID + HF_TOKEN) or MODEL_URL if configured.
# The script exits non-zero if it attempted a download and failed, causing the
# build/deploy to fail fast so you can correct env vars.
python3 scripts/ensure_model.py || (
    echo "ERROR: Model fetch failed. Set HF_REPO_ID/HF_TOKEN or MODEL_URL and retry." >&2
    exit 1
)
