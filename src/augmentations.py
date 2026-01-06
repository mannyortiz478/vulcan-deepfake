"""Simple waveform-level augmentations: additive noise, reverb, lowpass, random gain.
These are basic and fast; they operate on waveform tensors (1,d) and return augmented waveform.
"""
import random
import math
import torch
import torchaudio


def add_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Additive Gaussian noise to achieve target SNR in dB."""
    # waveform shape: (n_samples,) or (1, n_samples)
    if waveform.dim() > 1:
        wav = waveform.squeeze(0)
    else:
        wav = waveform

    sig_power = torch.mean(wav ** 2)
    snr = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr
    noise = torch.randn_like(wav) * torch.sqrt(noise_power + 1e-12)
    return (wav + noise).unsqueeze(0)


def apply_reverb(waveform: torch.Tensor, sample_rate: int, reverberance: int = 50) -> torch.Tensor:
    """Apply a simple reverb effect via sox 'reverb' effect."""
    try:
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, [["reverb", str(reverberance)]], channels_first=True
        )
        return augmented
    except Exception:
        return waveform


def apply_lowpass(waveform: torch.Tensor, sample_rate: int, cutoff: int = 7000) -> torch.Tensor:
    try:
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, [["lowpass", str(cutoff)]], channels_first=True
        )
        return augmented
    except Exception:
        return waveform


def random_gain(waveform: torch.Tensor, gain_dB_range: tuple = (-6, 6)) -> torch.Tensor:
    gain = random.uniform(*gain_dB_range)
    return waveform * (10 ** (gain / 20.0))


# New augmentations: pitch shift, time stretch, bandpass
def apply_pitch_shift(waveform: torch.Tensor, sample_rate: int, n_semitones: float = 2.0) -> torch.Tensor:
    """Pitch shift using sox 'pitch' effect. n_semitones can be negative."""
    try:
        # sox 'pitch' expects cents (100 * semitones)
        cents = float(n_semitones) * 100.0
        effects = [["pitch", str(cents)], ["rate", str(sample_rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects, channels_first=True)
        return augmented
    except Exception:
        return waveform


def apply_time_stretch(waveform: torch.Tensor, sample_rate: int, rate: float = 1.1) -> torch.Tensor:
    """Time stretch using sox 'tempo' effect (rate >1 speeds up, <1 slows down)."""
    try:
        effects = [["tempo", str(rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects, channels_first=True)
        return augmented
    except Exception:
        return waveform


def apply_bandpass(waveform: torch.Tensor, sample_rate: int, low_freq: int = 300, high_freq: int = 4000) -> torch.Tensor:
    """Bandpass via highpass then lowpass sox effects."""
    try:
        effects = [["highpass", str(low_freq)], ["lowpass", str(high_freq)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects, channels_first=True)
        return augmented
    except Exception:
        return waveform


def get_augment_transform(
    prob_noise: float = 0.5,
    snr_range: tuple = (5, 20),
    prob_reverb: float = 0.3,
    prob_lowpass: float = 0.3,
    prob_gain: float = 0.4,
    prob_pitch: float = 0.2,
    prob_time_stretch: float = 0.2,
    prob_bandpass: float = 0.2,
    pitch_range: tuple = (-2, 2),
    tempo_range: tuple = (0.9, 1.1),
    bandpass_range: tuple = (300, 4000),
):
    """Return a transform function that applies waveform-level augmentations.

    Usage: dataset = SomeDataset(transform=get_augment_transform())
    The transform should accept (waveform, sample_rate) and return a waveform.
    """

    def transform(waveform: torch.Tensor, sample_rate: int):
        out = waveform
        # Add noise
        if random.random() < prob_noise:
            snr = random.uniform(*snr_range)
            out = add_noise(out, snr)

        # Reverb
        if random.random() < prob_reverb:
            rev = random.randint(30, 70)
            out = apply_reverb(out, sample_rate, reverberance=rev)

        # Lowpass
        if random.random() < prob_lowpass:
            cutoff = random.randint(3000, min(8000, sample_rate // 2))
            out = apply_lowpass(out, sample_rate, cutoff=cutoff)

        # Random gain
        if random.random() < prob_gain:
            out = random_gain(out)

        # Pitch shift
        if random.random() < prob_pitch:
            semitone = random.uniform(*pitch_range)
            out = apply_pitch_shift(out, sample_rate, n_semitones=semitone)

        # Time stretch
        if random.random() < prob_time_stretch:
            rate = random.uniform(*tempo_range)
            out = apply_time_stretch(out, sample_rate, rate=rate)

        # Bandpass
        if random.random() < prob_bandpass:
            low = bandpass_range[0]
            high = bandpass_range[1]
            out = apply_bandpass(out, sample_rate, low_freq=low, high_freq=high)

        return out

    return transform
