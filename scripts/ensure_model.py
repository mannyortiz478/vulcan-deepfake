#!/usr/bin/env python3
"""Ensure the model file exists locally (download from Hugging Face or HTTP URL if needed).

Behavior:
- Checks for MODEL_PATH or the default target directory (/opt/render/project/src/models/).
- If model is missing and HF_REPO_ID is set, tries hf_hub_download (requires huggingface-hub).
- Else if MODEL_URL is set, downloads via HTTP(S) with retries and atomic write.
- Optional MODEL_SHA256 verifies checksum after download.
- Exits with 0 on success, non-zero on failure.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ensure_model")

# Optional dependencies
try:
    from huggingface_hub import hf_hub_download  # type: ignore
    HAS_HF = True
except Exception:
    hf_hub_download = None  # type: ignore
    HAS_HF = False

try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    HAS_REQUESTS = False


DEFAULT_BASENAME = "mesonet_whisper_mfcc_finetuned.pth"
DEFAULT_TARGET_DIR = Path("/opt/render/project/src/models")


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(temp_path: Path, final_path: Path) -> None:
    # Move into place atomically
    os.replace(str(temp_path), str(final_path))


def try_hf_download(repo_id: str, filename: str, token: Optional[str], target_dir: Path) -> Optional[Path]:
    if not HAS_HF:
        log.warning("huggingface-hub not installed; cannot download from Hugging Face Hub. Install it and retry.")
        return None
    log.info(f"Attempting Hugging Face download: repo_id={repo_id}, filename={filename}")
    try:
        p = hf_hub_download(repo_id=repo_id, filename=filename, token=token, local_dir=str(target_dir))
        log.info(f"Downloaded model from HF to: {p}")
        return Path(p)
    except Exception as e:
        log.exception(f"Hugging Face download failed: {e}")
        return None


def try_http_download(url: str, target_path: Path, attempts: int = 5, backoff: float = 2.0) -> Optional[Path]:
    if not HAS_REQUESTS:
        log.warning("requests package not installed; cannot download via HTTP. Install it and retry.")
        return None

    for attempt in range(1, attempts + 1):
        try:
            log.info(f"Downloading {url} (attempt {attempt}/{attempts})...")
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()

            target_dir = target_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(target_dir), delete=False) as tmpf:
                tmp_path = Path(tmpf.name)
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        tmpf.write(chunk)
                        downloaded += len(chunk)
                tmpf.flush()
            # Atomic move
            write_atomic(tmp_path, target_path)
            log.info(f"HTTP download complete: {target_path} ({downloaded} bytes)")
            return target_path
        except Exception as e:
            log.warning(f"Download attempt {attempt} failed: {e}")
            if attempt < attempts:
                sleep = backoff ** attempt
                log.info(f"Retrying in {sleep:.1f}s...")
                time.sleep(sleep)
            else:
                log.error("Maximum download attempts reached.")
    return None


def ensure_model() -> int:
    # Configuration via env
    model_path_env = os.environ.get("MODEL_PATH")
    model_basename = os.path.basename(model_path_env) if model_path_env else DEFAULT_BASENAME

    # If MODEL_PATH is absolute use it; else place in target dir
    default_target = Path(os.environ.get("MODEL_TARGET_DIR", str(DEFAULT_TARGET_DIR)))
    default_target.mkdir(parents=True, exist_ok=True)

    if model_path_env and os.path.isabs(model_path_env):
        candidates = [Path(model_path_env)]
    else:
        candidates = [default_target / model_basename]
        if model_path_env:
            candidates.insert(0, Path(model_path_env))

    # If one exists, validate optional checksum and succeed
    for c in candidates:
        if c.exists():
            log.info(f"Found existing model at {c}")
            sha_env = os.environ.get("MODEL_SHA256")
            if sha_env:
                actual = sha256_hex(c)
                if actual.lower() != sha_env.lower():
                    log.error(f"Checksum mismatch for existing file {c}: expected {sha_env}, got {actual}")
                    return 1
                log.info("Checksum OK")
            return 0

    # Not found; decide download method
    hf_repo = os.environ.get("HF_REPO_ID")
    hf_token = os.environ.get("HF_TOKEN")
    model_url = os.environ.get("MODEL_URL")
    auto_download = os.environ.get("MODEL_AUTO_DOWNLOAD", "1").lower() not in ("0", "false")

    if not auto_download:
        log.info("MODEL_AUTO_DOWNLOAD disabled; skipping download. Set MODEL_AUTO_DOWNLOAD=1 to enable.")
        return 2

    target_path = candidates[0]

    if hf_repo:
        # Hugging Face path tries first
        p = try_hf_download(hf_repo, model_basename, hf_token, default_target)
        if p and p.exists():
            # Verify checksum if present
            sha_env = os.environ.get("MODEL_SHA256")
            if sha_env:
                actual = sha256_hex(p)
                if actual.lower() != sha_env.lower():
                    log.error(f"Checksum mismatch after HF download: expected {sha_env}, got {actual}")
                    return 1
                log.info("Checksum OK")
            # Move to target_path if different
            if p != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(p, target_path)
            log.info(f"Model ready at {target_path}")
            return 0
        else:
            log.warning("Hugging Face download attempt did not produce a model file.")

    if model_url:
        p = try_http_download(model_url, target_path)
        if p and p.exists():
            sha_env = os.environ.get("MODEL_SHA256")
            if sha_env:
                actual = sha256_hex(p)
                if actual.lower() != sha_env.lower():
                    log.error(f"Checksum mismatch after HTTP download: expected {sha_env}, got {actual}")
                    return 1
                log.info("Checksum OK")
            log.info(f"Model ready at {target_path}")
            return 0
        else:
            log.error("HTTP download failed or did not produce the file.")

    log.error("No valid download method configured (set HF_REPO_ID + HF_TOKEN or MODEL_URL).")
    return 1


if __name__ == "__main__":
    rc = ensure_model()
    if rc == 0:
        log.info("Model check complete: FOUND")
        sys.exit(0)
    elif rc == 2:
        log.info("No download attempted (disabled or no source configured)")
        sys.exit(0)
    else:
        log.error("Model check failed. See logs for details.")
        sys.exit(1)
