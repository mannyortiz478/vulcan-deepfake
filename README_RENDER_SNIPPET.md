## Deploying on Render (quick checklist) ✅

If you're deploying this app on Render, follow these minimal steps to ensure the model is available at startup:

1. Upload model to Hugging Face (recommended)

   - Create a private Model repo on Hugging Face: https://huggingface.co/new
   - Upload `mesonet_whisper_mfcc_finetuned.pth` to that repo
   - Create a **Read** access token (Settings → Access Tokens)

2. Set the following Environment Variables in Render → Your Service → Environment:

   - `HF_REPO_ID`: your-hf-username/your-model-repo
   - `HF_TOKEN`: the HF read token (keep it secret)
   - `MODEL_AUTO_DOWNLOAD`: `1` (default) to enable auto-download at build/start
   - (Optional) `MODEL_SHA256`: hex sha256 checksum of the model to verify integrity
   - (Optional) `MODEL_PATH`: absolute path to place the model or a custom filename
   - (Optional) `MODEL_URL`: direct HTTPS URL if you prefer not to use HF

3. Ensure the build step installs required packages

   - `install.sh` now installs `huggingface-hub` and `requests` which are required for automatic downloads.

4. Deploy
   - The repository's `install.sh` calls `scripts/ensure_model.py` during setup and will fail fast if the model couldn't be fetched. Check Build logs and Service logs for download progress or failures.

Notes

- If you prefer, you can manually upload the model file to the service `models/` folder instead of using HF.
- For private repos, use `HF_TOKEN`. For public files, `MODEL_URL` is a simple fallback.
- Consider setting `MODEL_SHA256` to verify the file integrity after download.
