"""Minimal example showing how to finetune a pretrained wav2vec2 encoder.
Run:
    python examples/finetune_pretrained.py --asv_path /path/to/asv --epochs 3 --batch_size 4
"""
import argparse
from pathlib import Path

import torch
from src.datasets.detection_dataset import DetectionDataset
from src.models.pretrained import Wav2Vec2Classifier
from src.trainer import GDTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--freeze", action="store_true", help="Freeze encoder weights initially")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    ds = DetectionDataset(asvspoof_path=args.asv_path, subset="train")

    model = Wav2Vec2Classifier(model_name="facebook/wav2vec2-base-960h", freeze=args.freeze)
    model = model.to(device)

    trainer = GDTrainer(epochs=args.epochs, batch_size=args.batch_size, device=device)
    trainer.train(ds, model, test_len=0.2)
