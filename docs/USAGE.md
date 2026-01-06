# Using pretrained encoder and augmentations

You can optionally use a pretrained audio encoder (wav2vec2/HuBERT) and enable waveform + SpecAugment augmentations for training.

Install optional dependency:

```bash
pip install transformers
```

Example:

````python
from src.augmentations import get_augment_transform
from src.datasets.detection_dataset import DetectionDataset
from src.models.pretrained import Wav2Vec2Classifier
from src.trainer import GDTrainer

# dataset with augmentations enabled (default for subset='train')
train_ds = DetectionDataset(asvspoof_path='/path/to/asv', subset='train')

# or provide explicit transform:
train_ds = DetectionDataset(asvspoof_path='/path/to/asv', subset='train', transform=get_augment_transform())

# pretrained encoder (freeze encoder weights for faster convergence)
model = Wav2Vec2Classifier(model_name='facebook/wav2vec2-base-960h', freeze=True)

trainer = GDTrainer(epochs=10, batch_size=8, device='cpu')  # or 'cuda'
trainer.train(train_ds, model, test_len=0.2)

# Validation split and logging
You can instead request a stratified validation split via your YAML config by setting `data.val_split: 0.1` (keeps label balance).
Training will create a `logs/` folder next to saved checkpoints and produce `metrics.csv`. If TensorBoard is installed a TensorBoard run will also be written to the same folder.

# Config-driven augmentation
You can tune waveform augmentations directly from your YAML by adding an `augment` mapping under `data`. Example:

```yaml
data:
  val_split: 0.1
  augment:
    prob_noise: 0.5
    snr_range: [5, 20]
    prob_reverb: 0.2
    prob_lowpass: 0.2
    prob_gain: 0.3
    prob_pitch: 0.2
    pitch_range: [-2, 2]
    prob_time_stretch: 0.2
    tempo_range: [0.95, 1.05]
    prob_bandpass: 0.2
    bandpass_range: [300, 4000]
```

If `augment` is present, those parameters are used to create the augmentation transform and it's applied to the training dataset (validation/test remain unaugmented).

# Cross-validation (stratified k-fold)
You can run stratified k-fold cross-validation to evaluate hyperparameter choices more robustly. Use the provided script:

```bash
python scripts/cross_validate.py --asv_path /path/to/ASVspoof2021/DF --config configs/training/whisper_specrnet.yaml --folds 5 --out_dir cv_results --epochs 3
```

This will produce per-fold logs and a `summary.csv` with `fold` and `val_auc` columns in the `--out_dir` directory.```

Notes:

- `transformers` is optional; if not installed, using `Wav2Vec2Classifier` will raise a clear error message.
- Set `freeze=False` to fine-tune the encoder (requires more GPU memory).
````
