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


def test_deepfake_parser_fallback_matching(tmp_path):
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)

    # Protocol references sample by stem 'DF_E_3772106'
    lines = ["spk0 DF_E_3772106 dummy sys0 bonafide train pcm16 16"]
    (keys_dir / 'trial_metadata.txt').write_text('\n'.join(lines))

    # Flac file has extra prefix in stem
    partdir = base / 'ASVspoof2021_DF_eval_part00' / 'ASVspoof2021_DF_eval' / 'flac'
    partdir.mkdir(parents=True)
    (partdir / 'chunk_DF_E_3772106.flac').write_bytes(b'RIFF')

    ds = DeepFakeASVSpoofDataset(str(base), subset='train')
    df = ds.samples
    # Should find the file despite prefix mismatch
    assert len(df) == 1
    assert Path(df.iloc[0]['path']).exists()


def test_deepfake_parser_no_match_raises(tmp_path):
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)

    # Protocol references a sample that does not exist among flacs
    lines = ["spk0 MISSING_ID dummy sys0 bonafide train pcm16 16"]
    (keys_dir / 'trial_metadata.txt').write_text('\n'.join(lines))

    partdir = base / 'ASVspoof2021_DF_eval_part00' / 'ASVspoof2021_DF_eval' / 'flac'
    partdir.mkdir(parents=True)
    (partdir / 'some_other_file.flac').write_bytes(b'RIFF')

    try:
        ds = DeepFakeASVSpoofDataset(str(base), subset='train')
        # If no exception, fail
        assert False, "Expected KeyError for missing sample id"
    except KeyError as e:
        # message should list available stems for debugging
        assert 'Available stems' in str(e)


def test_get_file_references_top_level(tmp_path):
    # Some users have the top-level ASVspoof2021_DF_eval/ flac layout (no part folders)
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)

    lines = ["spk0 DF_E_3772106 dummy sys0 bonafide train pcm16 16"]
    (keys_dir / 'trial_metadata.txt').write_text('\n'.join(lines))

    partdir = base / 'ASVspoof2021_DF_eval' / 'flac'
    partdir.mkdir(parents=True)
    (partdir / 'DF_E_3772106.flac').write_bytes(b'RIFF')

    ds = DeepFakeASVSpoofDataset(str(base), subset='train')
    # ensure the dataset found the top-level flac dir
    assert 'DF_E_3772106' in ds.flac_paths
    df = ds.samples
    assert len(df) == 1
    assert Path(df.iloc[0]['path']).exists()


def test_no_flacs_found_informs_user(tmp_path):
    # When no flac files exist we raise a helpful FileNotFoundError
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)
    (keys_dir / 'trial_metadata.txt').write_text('spk0 DF_E_3772106 dummy sys0 bonafide train pcm16 16')

    try:
        DeepFakeASVSpoofDataset(str(base), subset='train')
        assert False, "Expected FileNotFoundError when no flac files are present"
    except FileNotFoundError as e:
        assert 'No .flac files found' in str(e)
