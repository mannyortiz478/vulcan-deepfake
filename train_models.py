import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml

from src.datasets.detection_dataset import DetectionDataset
from src.trainer import GDTrainer
from src.commons import set_seed
# note: import heavy model modules lazily inside train_nn to avoid import-time dependencies in tests


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


from src.utils.splits import stratified_split_df


def get_datasets(
    datasets_paths: List[Union[Path, str]],
    amount_to_use: Tuple[Optional[int], Optional[int]],
    val_split: float = 0.0,
    val_random_state: int = 42,
    augment_params: Optional[dict] = None,
) -> Tuple[DetectionDataset, Optional[DetectionDataset], DetectionDataset]:
    # Optionally create an augmentation transform from config params
    augment_transform = None
    if augment_params:
        from src.augmentations import get_augment_transform

        augment_transform = get_augment_transform(**augment_params)

    data_train = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        subset="train",
        reduced_number=amount_to_use[0],
        oversample=True,
        transform=augment_transform,
    )

    data_val = None
    if val_split and val_split > 0.0:
        try:
            train_df, val_df = stratified_split_df(
                data_train.samples, label_col="label", val_size=val_split, random_state=val_random_state
            )
            # Override samples for training set
            data_train.samples = train_df.reset_index(drop=True)
            # Build validation dataset and set its samples to the val split (no augmentation)
            data_val = DetectionDataset(
                asvspoof_path=datasets_paths[0],
                subset="val",
                reduced_number=None,
                oversample=False,
                augment=False,
            )
            data_val.samples = val_df.reset_index(drop=True)
        except Exception:
            data_val = None

    data_test = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        subset="test",
        reduced_number=amount_to_use[1],
        oversample=True,
    )

    return data_train, data_val, data_test


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    model_dir: Optional[Path] = None,
    amount_to_use: Tuple[Optional[int], Optional[int]] = (None, None),
    config_save_path: str = "configs",
    use_amp: Optional[bool] = None,
    accumulation_steps: int = 1,
    use_tqdm: bool = False,
) -> Tuple[str, str]:
    logging.info("Loading data...")
    model_config = config["model"]
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    timestamp = time.time()
    checkpoint_path = ""

    # optionally create validation split if requested in config
    val_split = config.get("data", {}).get("val_split", 0.0)
    val_random_state = config.get("data", {}).get("val_random_state", 42)

    # read augmentation params from config (optional)
    augment_params = config.get("data", {}).get("augment", None)

    data_train, data_val, data_test = get_datasets(
        datasets_paths=datasets_paths,
        amount_to_use=amount_to_use,
        val_split=val_split,
        val_random_state=val_random_state,
        augment_params=augment_params,
    )

    # Import models lazily to prevent heavy deps during tests
    from src.models import models

    current_model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    # pass use_amp via config if present
    use_amp = config.get('training', {}).get('use_amp', None)

    # If provided weights, apply corresponding ones (from an appropriate fold)
    model_path = config["checkpoint"]["path"]
    if model_path:
        current_model.load_state_dict(torch.load(model_path))
        logging.info(
            f"Finetuning '{model_name}' model, weights path: '{model_path}', on {len(data_train)} audio files."
        )
        if config["model"]["parameters"].get("freeze_encoder"):
            for param in current_model.whisper_model.parameters():
                param.requires_grad = False
    else:
        logging.info(f"Training '{model_name}' model on {len(data_train)} audio files.")
    current_model = current_model.to(device)

    use_scheduler = "rawnet3" in model_name.lower()

    # Prepare save destination and pass save path to trainer so best model is saved during training
    save_name = f"model__{model_name}__{timestamp}"
    save_dir = model_dir / save_name if model_dir is not None else None

    # TensorBoard / logging directory
    log_dir = str(save_dir / "logs") if save_dir is not None else None

    current_model = GDTrainer(
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_kwargs=optimizer_config,
        use_scheduler=use_scheduler,
        use_amp=use_amp,
        accumulation_steps=accumulation_steps,
        use_tqdm=use_tqdm,
        log_dir=log_dir,
    ).train(
        dataset=data_train,
        model=current_model,
        val_dataset=data_val,
        test_dataset=data_test,
        save_path=str(save_dir / "ckpt.pth") if save_dir is not None else None,
    )

    # Final model save already handled during training; ensure checkpoint path resolved
    if model_dir is not None:
        checkpoint_path = str(save_dir.resolve() / "ckpt.pth")

    # (old behaviour replaced by saving during training)

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"path": checkpoint_path}
        config_name = f"model__{model_name}__{timestamp}.yaml"
        config_save_path = str(Path(config_save_path) / config_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        logging.info("Test config saved at location '{}'!".format(config_save_path))
    return config_save_path, checkpoint_path


def main(args):
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    # Prefer explicit device if provided, otherwise honor --cpu flag and availability
    if args.device is not None:
        device = args.device
    # Prefer explicit device if provided, otherwise honor --cpu flag and availability
    if args.device is not None:
        device = args.device
    elif not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Respect explicit amp and accumulation request, otherwise let trainer auto-detect.
    use_amp = args.amp if hasattr(args, 'amp') else None
    accum_steps = args.accum_steps if hasattr(args, 'accum_steps') else 1

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_nn(
        datasets_paths=[
            args.asv_path,
            args.wavefake_path,
            args.celeb_path,
            args.asv19_path,
        ],
        device=device,
        amount_to_use=(args.train_amount, args.test_amount),
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH = "../datasets/ASVspoof2021/DF"

    parser.add_argument(
        "--asv_path",
        type=str,
        default=ASVSPOOF_DATASET_PATH,
        help="Path to ASVspoof2021 dataset directory",
    )

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

    default_train_amount = None
    parser.add_argument(
        "--train_amount",
        "-a",
        help=f"Amount of files to load for training.",
        type=int,
        default=default_train_amount,
    )

    default_test_amount = None
    parser.add_argument(
        "--test_amount",
        "-ta",
        help=f"Amount of files to load for testing.",
        type=int,
        default=default_test_amount,
    )

    default_batch_size = 8
    parser.add_argument(
        "--batch_size",
        "-b",
        help=f"Batch size (default: {default_batch_size}).",
        type=int,
        default=default_batch_size,
    )

    default_epochs = 10
    parser.add_argument(
        "--epochs",
        "-e",
        help=f"Epochs (default: {default_epochs}).",
        type=int,
        default=default_epochs,
    )

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt",
        help=f"Checkpoint directory (default: {default_model_dir}).",
        type=str,
        default=default_model_dir,
    )

    parser.add_argument("--cpu", "-c", help="Force using cpu?", action="store_true")
    parser.add_argument("--device", help="Explicit device string (e.g. cpu, cuda, cuda:0)", type=str, default=None)
    parser.add_argument("--amp", help="Enable mixed precision (automatic), default: auto when cuda available", action="store_true")
    parser.add_argument("--accum-steps", help="Gradient accumulation steps (default: 1)", type=int, default=1)
    parser.add_argument("--use-tqdm", help="Show tqdm progress bar during training", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
