"""Run stratified k-fold cross-validation for robust hyperparameter selection.

Example:
  python scripts/cross_validate.py --asv_path /path/to/ASVspoof2021/DF --config configs/training/whisper_specrnet.yaml --folds 5 --out_dir cv_results --epochs 3

Notes:
- Requires configuration compatible with `train_models.train_nn` (model name + parameters)
- Each fold will create logs under `out_dir/fold_<i>/logs` and a metrics CSV; final summary will be written to `out_dir/summary.csv`.
"""
import argparse
import logging
from pathlib import Path
import sys
import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path so `src` imports work when this script is executed directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.utils.cv import stratified_kfold_df


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="cv_results")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for number of epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_fake_audio", action="store_true", help="Do not load audio files; use random tensors (useful for smoke tests)")
    parser.add_argument("--dry_run", action="store_true", help="Do not train models; only prepare folds and write summary (useful for smoke tests)")
    return parser.parse_args()


def read_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_best_val_auc_from_log(logdir: Path):
    csv_path = logdir / "metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "val_auc" in df.columns:
        return float(df["val_auc"].max())
    return None


def build_dataset_from_df(base_path: str, df: pd.DataFrame, subset: str = "train", augment: bool = False):
    ds = DeepFakeASVSpoofDataset(base_path, subset=subset)
    ds.samples = df.reset_index(drop=True)
    # force augmentation flag
    if not augment:
        ds.transform = None
    return ds


def main():
    args = parse_args()
    cfg = read_config(args.config)

    # read full training set as DF using dataset reader
    full_ds = DeepFakeASVSpoofDataset(args.asv_path, subset="train")
    df = full_ds.samples

    folds = stratified_kfold_df(df, label_col="label", n_splits=args.folds, random_state=cfg.get("data", {}).get("seed", 42))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # optionally replace audio loader with fake loader for smoke tests / CI
    if args.use_fake_audio:
        import torchaudio
        from src.datasets import base_dataset as base_ds

        def _fake_load(path, normalize=True):
            import torch

            return torch.randn(1, 16000), 16000

        # Override audio loading and preprocessing steps to avoid optional deps in CI
        torchaudio.load = _fake_load
        base_ds.APPLY_TRIMMING = False
        base_ds.APPLY_PADDING = False
        base_ds.SOX_SILENCE = []  # no sox effects

    for i, (train_df, val_df) in enumerate(folds):
        LOGGER.info(f"Starting fold {i+1}/{len(folds)}: train={len(train_df)}, val={len(val_df)}")

        train_ds = build_dataset_from_df(args.asv_path, train_df, subset="train", augment=True)
        val_ds = build_dataset_from_df(args.asv_path, val_df, subset="val", augment=False)

        # lazy import models and trainer
        from src.models import models
        from src.trainer import GDTrainer

        model_name = cfg["model"]["name"]
        model_parameters = cfg["model"]["parameters"]

        model = models.get_model(model_name=model_name, config=model_parameters, device=("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu"))

        # optional epochs override
        epochs = args.epochs if args.epochs is not None else cfg.get("training", {}).get("epochs", cfg.get("training", {}).get("num_epochs", 10))

        save_dir = out_dir / f"fold_{i+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir = save_dir / "logs"

        if args.dry_run:
            # Skip actual training - useful for quick smoke tests
            LOGGER.info(f"Dry run enabled - skipping training for fold {i+1}")
            results.append({"fold": i + 1, "val_auc": None})
            continue

        trainer = GDTrainer(
            epochs=epochs,
            batch_size=args.batch_size,
            device=("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu"),
            optimizer_kwargs=cfg["model"].get("optimizer", {}),
            use_scheduler=False,
            use_amp=False,
            accumulation_steps=1,
            use_tqdm=False,
            log_dir=str(log_dir),
        )

        trainer.train(dataset=train_ds, model=model, val_dataset=val_ds, save_path=str(save_dir / "ckpt.pth"))

        best_val_auc = read_best_val_auc_from_log(log_dir)
        results.append({"fold": i + 1, "val_auc": best_val_auc})

    # write summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "summary.csv", index=False)
    LOGGER.info(f"Cross-validation complete. Summary written to {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
