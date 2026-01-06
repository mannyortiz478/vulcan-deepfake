"""A generic training wrapper."""
from copy import deepcopy
import logging
from typing import Callable, List, Optional
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
        use_amp: Optional[bool] = None,
        accumulation_steps: int = 1,
        use_tqdm: bool = False,
        log_dir: Optional[str] = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler
        # Auto-enable AMP when not on CPU unless explicitly disabled
        if use_amp is None:
            self.use_amp = (self.device != "cpu") and torch.cuda.is_available()
        else:
            self.use_amp = use_amp
        # Gradient accumulation steps
        self.accumulation_steps = max(1, int(accumulation_steps))
        # progress bar
        self.use_tqdm = bool(use_tqdm)
        # logging dir for TensorBoard / CSV
        self.log_dir = log_dir


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):
    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        test_len: Optional[float] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        save_path: Optional[str] = None,
    ):
        # Decide on train/validation/test datasets. Prefer explicit val_dataset for validation.
        train = dataset
        validation = val_dataset if val_dataset is not None else test_dataset

        # If validation isn't provided, fall back to splitting the provided dataset according to test_len
        if validation is None and test_len is not None:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, validation = torch.utils.data.random_split(dataset, lengths)

        num_workers = 0 if self.device == 'cpu' else 6

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            validation,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        # Setup logging (TensorBoard + CSV) if requested
        writer = None
        csv_path = None
        if self.log_dir is not None:
            import os
            import csv
            os.makedirs(self.log_dir, exist_ok=True)
            csv_path = os.path.join(self.log_dir, "metrics.csv")
            # write csv header
            with open(csv_path, "w", newline="") as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "val_loss",
                        "val_acc",
                        "val_auc",
                        "val_eer",
                        "throughput",
                    ]
                )

            # Try to init TensorBoard writer if installed, otherwise keep it None
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(self.log_dir)
            except Exception:
                writer = None

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0
        best_auc = 0.0

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=5e-6,
                # verbose=True,
            )
        use_cuda = self.device != "cpu"

        # AMP scaler (if enabled)
        scaler = None
        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            # Compute and log accumulation-related info
            steps_per_epoch = int(np.ceil(len(train) / float(self.batch_size))) if len(train) > 0 else 0
            optimizer_steps_per_epoch = int(np.ceil(steps_per_epoch / float(self.accumulation_steps))) if steps_per_epoch > 0 else 0
            effective_batch = int(self.batch_size * self.accumulation_steps)
            LOGGER.info(
                f"Accumulation: steps={self.accumulation_steps}, effective_batch_size={effective_batch}, "
                f"train_steps={steps_per_epoch}, optimizer_steps_per_epoch={optimizer_steps_per_epoch}"
            )

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            # zero grads before starting accumulation cycle
            optim.zero_grad()

            global_opt_step = 0

            # epoch timing
            epoch_start_time = time.time()

            # optionally use tqdm for progress
            if self.use_tqdm:
                iter_train = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch}")
            else:
                iter_train = train_loader

            for i, (batch_x, _, batch_y) in enumerate(iter_train):
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                # compute batch loss
                if self.use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        batch_out, batch_loss = forward_and_loss_fn(
                            model, criterion, batch_x, batch_y, use_cuda=use_cuda
                        )
                else:
                    batch_out, batch_loss = forward_and_loss_fn(
                        model, criterion, batch_x, batch_y, use_cuda=use_cuda
                    )

                # Use sigmoid scores and a clear 0.5 threshold
                batch_scores = torch.sigmoid(batch_out)
                batch_pred = (batch_scores >= 0.5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                running_loss += batch_loss.item() * batch_size

                # update tqdm postfix if used
                if self.use_tqdm:
                    try:
                        iter_train.set_postfix({'loss': running_loss/num_total, 'acc': num_correct/num_total*100})
                    except Exception:
                        pass

                if i % 100 == 0 and not self.use_tqdm:
                    LOGGER.info(
                        f"[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct/num_total*100}"
                    )

                # scale loss for accumulation and backprop
                loss_to_backprop = batch_loss / float(self.accumulation_steps)

                if self.use_amp and scaler is not None:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                # perform optimizer step every accumulation_steps
                if (i + 1) % self.accumulation_steps == 0:
                    if self.use_amp and scaler is not None:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()

                    # zero grads after step
                    optim.zero_grad()

                    if self.use_scheduler:
                        scheduler.step()

                    # increment and occasionally log optimizer-global steps
                    global_opt_step += 1
                    if global_opt_step % 50 == 0:
                        LOGGER.info(f"Optimizer step: {global_opt_step} (epoch {epoch})")

            # handle leftover gradients if dataset size not divisible by accumulation_steps
            if (i + 1) % self.accumulation_steps != 0:
                if self.use_amp and scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()

                optim.zero_grad()
                if self.use_scheduler:
                    scheduler.step()

            # close tqdm if used
            if self.use_tqdm and hasattr(iter_train, 'close'):
                try:
                    iter_train.close()
                except Exception:
                    pass

            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

            # Epoch throughput and summary
            elapsed = time.time() - epoch_start_time if 'epoch_start_time' in locals() else 0.0
            total_train_samples = len(train)
            throughput = total_train_samples / elapsed if elapsed > 0 else 0.0

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}, train/accuracy: {train_accuracy}, optimizer_steps: {global_opt_step}, throughput: {throughput:.2f} samples/s"
            )

            val_running_loss = 0.0
            all_labels = []
            all_scores = []
            num_total = 0
            model.eval()

            for batch_x, _, batch_y in val_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                with torch.no_grad():
                    batch_logits = model(batch_x)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                batch_loss = criterion(batch_logits, batch_y)

                val_running_loss += batch_loss.item() * batch_size

                batch_scores = torch.sigmoid(batch_logits).detach().cpu().numpy().ravel()
                all_scores.extend(batch_scores.tolist())
                all_labels.extend(batch_y.detach().cpu().numpy().ravel().tolist())

            if num_total == 0:
                num_total = 1

            val_running_loss /= num_total

            # Compute AUC and EER safely
            try:
                if len(set(all_labels)) > 1:
                    val_auc = float(roc_auc_score(all_labels, all_scores))
                    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
                    fnr = 1 - tpr
                    eer_idx = np.nanargmin(np.abs(fnr - fpr))
                    val_eer = float(fpr[eer_idx])
                else:
                    val_auc = 0.5
                    val_eer = 1.0
            except Exception:
                val_auc = 0.5
                val_eer = 1.0

            val_pred_labels = (np.array(all_scores) >= 0.5).astype(int)
            val_acc = 100 * (val_pred_labels == np.array(all_labels)).mean()

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: val/loss: {val_running_loss}, val/accuracy: {val_acc:.3f}, val/auc: {val_auc:.4f}, val/eer: {val_eer:.4f}"
            )

            # Model selection by AUC (more robust for imbalanced tasks)
            if best_model is None or val_auc > best_auc:
                best_auc = val_auc
                best_acc = val_acc
                best_model = deepcopy(model.state_dict())
                # optionally save best model to disk if path provided
                if save_path is not None:
                    torch.save(best_model, save_path)

            LOGGER.info(
                f"[{epoch:04d}]: {running_loss} - train acc: {train_accuracy} - val_acc: {val_acc}"
            )

            # Logging to TensorBoard/CSV if enabled
            if writer is not None:
                writer.add_scalar("train/loss", running_loss, epoch)
                writer.add_scalar("train/accuracy", train_accuracy, epoch)
                writer.add_scalar("val/loss", val_running_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/auc", val_auc, epoch)
                writer.add_scalar("val/eer", val_eer, epoch)

            if csv_path is not None:
                import csv
                with open(csv_path, "a", newline="") as csvfile:
                    csvw = csv.writer(csvfile)
                    csvw.writerow([epoch, running_loss, train_accuracy, val_running_loss, val_acc, val_auc, val_eer, throughput])

        model.load_state_dict(best_model)

        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass

        return model
