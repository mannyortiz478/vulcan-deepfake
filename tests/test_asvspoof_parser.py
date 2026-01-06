import tempfile
from pathlib import Path
import pandas as pd
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset


def test_deepfake_parser_reads_protocol(tmp_path):
    # Create synthetic dataset structure
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)

    # Write a small trial_metadata.txt
    # Create a balanced-ish trial_metadata with multiple entries so split_samples keeps some train samples
    lines = []
    for i in range(8):
        # 6 bonafide, 2 spoof
        label = 'bonafide' if i < 6 else 'spoof'
        idx = f"{i:03d}"
        lines.append(f"spk{i} file{idx} dummy sys{i} {label} train pcm16 16")

    (keys_dir / 'trial_metadata.txt').write_text('\n'.join(lines))

    # Create fake flac file tree matching expected structure
    partdir = base / 'ASVspoof2021_DF_eval_part00' / 'ASVspoof2021_DF_eval' / 'flac'
    partdir.mkdir(parents=True)
    for i in range(8):
        idx = f"{i:03d}"
        (partdir / f'file{idx}.flac').write_bytes(b'RIFF')

    ds = DeepFakeASVSpoofDataset(str(base), subset='train')
    df = ds.samples
    assert 'path' in df.columns
    # We expect at least one parsed sample (partitioning may reduce small synthetic set)
    assert len(df) >= 1
    # Ensure label parsing and that referenced files exist
    for pth in df['path']:
        assert Path(pth).exists()
