"""Quick evaluation script to run a trained model on ASVspoof2021 DF trial set and print AUC/EER.
Usage example:
    python scripts/evaluate_asvspoof.py --asv_path /path/to/ASVspoof2021/DF --ckpt trained_models/.../ckpt.pth --config configs/training/whisper_specrnet.yaml --batch_size 8 --cpu
"""
import argparse
import logging
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.models import models


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    model_parameters = config["model"]["parameters"]

    model = models.get_model(model_name=model_name, config=model_parameters, device=device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    ds = DeepFakeASVSpoofDataset(args.asv_path, subset="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_labels = []
    all_scores = []

    for batch_x, _, batch_y in loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            logits = model(batch_x)
            scores = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
            all_scores.extend(scores)
            all_labels.extend(batch_y.numpy().ravel().tolist())

    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
        fpr, tpr, thr = roc_curve(all_labels, all_scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float(fpr[eer_idx])
    else:
        auc = 0.5
        eer = 1.0

    LOGGER.info(f"AUC: {auc:.4f}, EER: {eer:.4f}")


if __name__ == "__main__":
    main()
