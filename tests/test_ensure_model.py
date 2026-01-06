import os
import tempfile
from pathlib import Path
import hashlib
import importlib.util

import pytest


def load_module():
    # Load scripts/ensure_model.py as a fresh module to allow monkeypatching module attrs
    path = Path(__file__).resolve().parents[1] / "scripts" / "ensure_model.py"
    spec = importlib.util.spec_from_file_location("ensure_model_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def test_existing_model_ok(tmp_path, monkeypatch):
    mod = load_module()
    target = tmp_path / "mesonet_whisper_mfcc_finetuned.pth"
    write_file(target, b"hello-world")

    monkeypatch.setenv("MODEL_PATH", str(target))
    rc = mod.ensure_model()
    assert rc == 0


def test_auto_download_disabled(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("MODEL_TARGET_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_AUTO_DOWNLOAD", "0")
    # Ensure no model present
    rc = mod.ensure_model()
    assert rc == 2


def test_checksum_mismatch(tmp_path, monkeypatch):
    mod = load_module()
    target = tmp_path / "mesonet_whisper_mfcc_finetuned.pth"
    write_file(target, b"bad-data")

    wrong = "deadbeef"
    monkeypatch.setenv("MODEL_TARGET_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_SHA256", wrong)

    rc = mod.ensure_model()
    assert rc == 1


def test_hf_download_success(tmp_path, monkeypatch):
    mod = load_module()

    # Monkeypatch the huggingface helper to simulate a successful download
    def fake_hf_hub_download(repo_id, filename, token, local_dir):
        p = Path(local_dir) / filename
        write_file(p, b"hf-model-contents")
        return str(p)

    monkeypatch.setattr(mod, "HAS_HF", True)
    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_REPO_ID", "user/repo")
    monkeypatch.setenv("MODEL_TARGET_DIR", str(tmp_path))

    rc = mod.ensure_model()
    assert rc == 0
    target = tmp_path / "mesonet_whisper_mfcc_finetuned.pth"
    assert target.exists()
    assert target.read_bytes() == b"hf-model-contents"


def test_http_download_success(tmp_path, monkeypatch):
    mod = load_module()

    # Fake requests behavior
    class FakeResp:
        def __init__(self, data: bytes):
            self._data = data
            self._iter = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield self._data

    class FakeRequests:
        def get(self, url, stream=True, timeout=30):
            return FakeResp(b"http-model")

    monkeypatch.setattr(mod, "HAS_REQUESTS", True)
    monkeypatch.setattr(mod, "requests", FakeRequests())

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_URL", "https://example.com/model.pth")
    monkeypatch.setenv("MODEL_TARGET_DIR", str(tmp_path))

    rc = mod.ensure_model()
    assert rc == 0
    target = tmp_path / "mesonet_whisper_mfcc_finetuned.pth"
    assert target.exists()
    assert target.read_bytes() == b"http-model"


def test_hf_checksum_verification(tmp_path, monkeypatch):
    mod = load_module()

    # Create HF downloader that returns a file with known contents
    contents = b"verified-content"
    def fake_hf_hub_download(repo_id, filename, token, local_dir):
        p = Path(local_dir) / filename
        write_file(p, contents)
        return str(p)

    monkeypatch.setattr(mod, "HAS_HF", True)
    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)

    sha = sha256_hex_bytes(contents)

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_REPO_ID", "user/repo")
    monkeypatch.setenv("MODEL_TARGET_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_SHA256", sha)

    rc = mod.ensure_model()
    assert rc == 0
    target = tmp_path / "mesonet_whisper_mfcc_finetuned.pth"
    assert target.exists()
    assert sha256_hex_bytes(target.read_bytes()) == sha
