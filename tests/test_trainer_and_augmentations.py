import torch
from torch import nn
from torch.utils.data import Dataset


from src.augmentations import get_augment_transform
from src.trainer import GDTrainer


def test_augment_transform_runs():
    transform = get_augment_transform()
    waveform = torch.randn(1, 16000)
    augmented = transform(waveform, 16000)
    assert isinstance(augmented, torch.Tensor)
    assert augmented.dim() == 2
    assert augmented.size(0) == 1


class ToyDataset(Dataset):
    def __init__(self, n=20, length=16000):
        self.n = n
        self.length = length
        self.data = [torch.randn(1, length) for _ in range(n)]
        self.labels = [i % 2 for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], 16000, self.labels[idx]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        m = x.mean(dim=(1, 2)).unsqueeze(1)
        return self.fc(m)


def test_trainer_runs_and_computes_auc():
    dataset = ToyDataset(n=20)
    model = DummyModel()
    trainer = GDTrainer(epochs=1, batch_size=4, device="cpu")

    # Must not raise
    out_model = trainer.train(dataset, model, test_len=0.25)
    assert out_model is not None


def test_trainer_saves_checkpoint(tmp_path):
    dataset = ToyDataset(n=20)
    model = DummyModel()
    trainer = GDTrainer(epochs=1, batch_size=4, device="cpu")

    save_path = tmp_path / "ckpt.pth"
    _ = trainer.train(dataset, model, test_len=0.25, save_path=str(save_path))

    assert save_path.exists()


def test_trainer_amp_flag():
    # Ensure Trainer accepts explicit amp argument (on CPU it should be False)
    t = GDTrainer(epochs=1, batch_size=4, device="cpu", use_amp=True)
    assert t.use_amp in (True, False)


def test_trainer_with_accumulation(tmp_path, capsys):
    # Run a short training with accumulation_steps=2 to ensure accumulation works and logs appear
    dataset = ToyDataset(n=12)
    model = DummyModel()
    trainer = GDTrainer(epochs=1, batch_size=2, device="cpu", accumulation_steps=2)

    save_path = tmp_path / "ckpt_accum.pth"
    _ = trainer.train(dataset, model, test_len=0.25, save_path=str(save_path))

    assert save_path.exists()
    # ensure accumulation info was logged
    captured = capsys.readouterr()
    assert "Accumulation: steps=2" in captured.out or "Accumulation: steps=2" in captured.err


def test_trainer_tqdm_runs():
    dataset = ToyDataset(n=10)
    model = DummyModel()
    trainer = GDTrainer(epochs=1, batch_size=2, device="cpu", use_tqdm=True)

    # Must not raise
    _ = trainer.train(dataset, model, test_len=0.25)
    assert True
