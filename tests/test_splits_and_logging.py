import os
import tempfile
import torch
import pandas as pd
from pathlib import Path
from src.utils.splits import stratified_split_df
from src.augmentations import apply_pitch_shift, apply_time_stretch, apply_bandpass
from src.trainer import GDTrainer


def test_stratified_split_preserves_ratio():
    # Create imbalanced df: 80 bonafide, 20 spoof
    n_b = 80
    n_s = 20
    df = pd.DataFrame({
        "path": [f"/fake/path/{i}.wav" for i in range(n_b + n_s)],
        "label": ["bonafide"] * n_b + ["spoof"] * n_s,
        "attack_type": ["N/A"] * (n_b + n_s),
    })

    train_df, val_df = stratified_split_df(df, label_col="label", val_size=0.2, random_state=123)

    # Check proportions roughly preserved
    def ratio(d):
        counts = d["label"].value_counts(normalize=True)
        return counts.get("spoof", 0.0)

    orig_ratio = ratio(df)
    val_ratio = ratio(val_df)
    assert abs(orig_ratio - val_ratio) < 0.05


def test_waveform_augments_do_not_crash():
    waveform = torch.randn(1, 16000)
    sr = 16000

    out1 = apply_pitch_shift(waveform, sr, n_semitones=1.0)
    assert isinstance(out1, torch.Tensor)

    out2 = apply_time_stretch(waveform, sr, rate=1.05)
    assert isinstance(out2, torch.Tensor)

    out3 = apply_bandpass(waveform, sr, low_freq=300, high_freq=4000)
    assert isinstance(out3, torch.Tensor)


def test_get_augment_transform_forced():
    from src.augmentations import get_augment_transform

    waveform = torch.randn(1, 16000)
    sr = 16000

    transform = get_augment_transform(
        prob_noise=0.0,
        prob_reverb=0.0,
        prob_lowpass=0.0,
        prob_gain=0.0,
        prob_pitch=1.0,
        prob_time_stretch=1.0,
        prob_bandpass=1.0,
        pitch_range=(1.0, 1.0),
        tempo_range=(1.01, 1.01),
        bandpass_range=(300, 4000),
    )

    out = transform(waveform, sr)
    assert isinstance(out, torch.Tensor)

def _toy_dataset(num=8):
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, num):
            self.num = num

        def __len__(self):
            return self.num

        def __getitem__(self, idx):
            x = torch.randn(1, 16000)
            y = 1 if idx % 2 == 0 else 0
            return x, 16000, y

    return ToyDataset(num)


def test_trainer_logging_creates_files(tmp_path):
    train_ds = _toy_dataset(16)
    val_ds = _toy_dataset(8)

    # simple model that consumes waveform and outputs 1 logit; reuse existing model pattern
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16000, 1)

        def forward(self, x):
            # x is (batch, 1, samples)
            x = x.view(x.size(0), -1)
            return self.lin(x)

    model = ToyModel()

    trainer = GDTrainer(epochs=1, batch_size=4, device="cpu", accumulation_steps=1, use_tqdm=False, log_dir=str(tmp_path))
    trainer.train(dataset=train_ds, model=model, val_dataset=val_ds, save_path=None)

    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.exists()
    content = metrics_csv.read_text()
    assert "epoch" in content


def test_get_datasets_augments(tmp_path):
    # Create temp ASV structure
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)

    # minimal trial file
    lines = [f"spk{i} file{i:03d} dummy sys{i} bonafide train pcm16 16" for i in range(50)]
    (keys_dir / 'trial_metadata.txt').write_text('\n'.join(lines))

    partdir = base / 'ASVspoof2021_DF_eval_part00' / 'ASVspoof2021_DF_eval' / 'flac'
    partdir.mkdir(parents=True)
    for i in range(50):
        (partdir / f'file{i:03d}.flac').write_bytes(b'RIFF')

    from train_models import get_datasets

    augment_params = {"prob_pitch": 1.0, "pitch_range": (1.0, 1.0)}
    train_ds, val_ds, test_ds = get_datasets([str(base)], (None, None), val_split=0.2, val_random_state=1, augment_params=augment_params)

    assert train_ds.transform is not None and callable(train_ds.transform)


def test_cross_validate_smoke(tmp_path):
    # Smoke-run the cross-validate script with a small synthetic dataset (no heavy deps)
    base = Path(tmp_path)
    keys_dir = base / 'keys' / 'CM'
    keys_dir.mkdir(parents=True)

    lines = []
    for i in range(20):
        label = 'bonafide' if i % 4 != 0 else 'spoof'
        lines.append(f"spk{i} file{i:03d} dummy sys{i} {label} train pcm16 16")
    (keys_dir / 'trial_metadata.txt').write_text('\n'.join(lines))

    partdir = base / 'ASVspoof2021_DF_eval_part00' / 'ASVspoof2021_DF_eval' / 'flac'
    partdir.mkdir(parents=True)
    for i in range(20):
        (partdir / f'file{i:03d}.flac').write_bytes(b'RIFF')

    # Create a minimal config
    cfg = {
        'model': {'name': 'lcnn', 'parameters': {'input_channels': 1, 'frontend_algorithm': ['lfcc']}, 'optimizer': {'lr': 1e-3}},
        'data': {'seed': 42}
    }
    cfg_path = Path(tmp_path) / 'cfg.yaml'
    import yaml
    cfg_path.write_text(yaml.dump(cfg))

    # Run cross_validate with 2 folds and one epoch to keep it fast
    import subprocess, sys
    out_dir = Path(tmp_path) / 'cv_out'
    cmd = [
        sys.executable, 'scripts/cross_validate.py',
        '--asv_path', str(base), '--config', str(cfg_path), '--folds', '2', '--out_dir', str(out_dir), '--epochs', '1', '--batch_size', '4', '--cpu', '--use_fake_audio', '--dry_run'
    ]
    # run as a subprocess to simulate user invocation
    subprocess.check_call(cmd)

    summary = out_dir / 'summary.csv'
    assert summary.exists()
