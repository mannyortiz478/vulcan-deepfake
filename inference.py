import torch
import librosa
import numpy as np
import torch.nn.functional as F
from src.models.whisper_meso_net import WhisperMesoNet
import src.models.whisper_main as wm

class MesoNetDetector:
    def __init__(self, model_path, config_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure 2 channels as required by the model weights
        self.model = WhisperMesoNet(input_channels=2, freeze_encoder=True, device=self.device) 
        
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        audio_path,
        window_sec: float = 10.0,
        hop_sec: float = 5.0,
        max_windows: int = 8,
        normalization: str = "mean",  # options: 'mean', 'cmvn', 'global'
        augment: bool = False,
        augment_noise_db: list | None = None,
        augment_lowpass_cutoffs: list | None = None,
        top_k: int | None = 3,
        top_pct: float = 0.2,
        consistency_threshold: float = 0.5,
        consistency_min_count: int = 2,
        return_top_k: int = 3,
    ):
        try:
            with torch.no_grad():
                sr = 16000
                audio, _ = librosa.load(audio_path, sr=sr)
                audio = librosa.util.normalize(audio)

                audio_t = torch.from_numpy(audio).to(self.device).float()
                win = int(window_sec * sr)
                hop = int(hop_sec * sr)

                # If file shorter than one window, just pad one window
                if audio_t.numel() < win:
                    audio_t = F.pad(audio_t, (0, win - audio_t.numel()))
                    starts = [0]
                else:
                    starts = list(range(0, audio_t.numel() - win + 1, hop))

                # Downsample number of windows to max_windows (evenly spaced)
                if len(starts) > max_windows:
                    idxs = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
                    starts = [starts[i] for i in idxs]

                # Setup augmentation defaults
                if augment_noise_db is None:
                    augment_noise_db = [25]
                if augment_lowpass_cutoffs is None:
                    augment_lowpass_cutoffs = [7000]

                # helper: add noise at given SNR (dB)
                def add_noise_tensor(sig: torch.Tensor, snr_db: float) -> torch.Tensor:
                    rms_signal = sig.norm(p=2) / (sig.numel() ** 0.5)
                    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
                    noise = torch.randn_like(sig) * rms_noise
                    return sig + noise

                # helper: lowpass via torchaudio biquad filter (accept mono tensor)
                def lowpass_tensor(sig: torch.Tensor, cutoff: float) -> torch.Tensor:
                    try:
                        import torchaudio.functional as TAF
                        filtered = TAF.lowpass_biquad(sig.unsqueeze(0), sr, cutoff).squeeze(0)
                        return filtered
                    except Exception:
                        # fallback: return original if lowpass not available
                        return sig

                window_probs = []  # per-window aggregated (max across augs) prob
                window_logits = []  # per-window aggregated (max across augs) logit
                window_aug_probs = []  # list of lists of augmentation probs per window
                window_aug_logits = []

                # Precompute transforms
                window = torch.hann_window(1024).to(self.device)
                mel_basis = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=80)
                mel_basis = torch.from_numpy(mel_basis).to(self.device).float()

                for s in starts:
                    seg = audio_t[s : s + win]
                    if seg.numel() < win:
                        seg = F.pad(seg, (0, win - seg.numel()))

                    segs_to_eval = [seg]
                    if augment:
                        for snr in augment_noise_db:
                            segs_to_eval.append(add_noise_tensor(seg, snr))
                        for cutoff in augment_lowpass_cutoffs:
                            segs_to_eval.append(lowpass_tensor(seg, cutoff))

                    aug_probs = []
                    aug_logits = []

                    for seg_eval in segs_to_eval:
                        stft = torch.stft(seg_eval, n_fft=1024, hop_length=160,
                                          window=window, center=True, return_complex=True)
                        magnitudes = stft.abs() ** 2

                        mel = mel_basis @ magnitudes
                        mel = torch.log10(torch.clamp(mel, min=1e-10))

                        # normalize + crop/pad frames to 3000
                        mel = (mel + 4.0) / 4.0

                        # Normalization strategy
                        if normalization == "mean":
                            mean = mel.mean(dim=1, keepdim=True)
                            mel = mel - mean
                        elif normalization == "cmvn":
                            eps = 1e-5
                            mean = mel.mean(dim=1, keepdim=True)
                            std = mel.std(dim=1, keepdim=True, unbiased=False)
                            mel = (mel - mean) / (std + eps)
                        elif normalization == "global":
                            mel = (mel - mel.mean()) / (mel.std(unbiased=False) + 1e-5)
                        else:
                            raise ValueError(f"Unsupported normalization: {normalization}")

                        mel = mel.clamp(-5.0, 5.0)

                        # Force fixed time dimension (3000)
                        T = mel.shape[1]
                        if T < 3000:
                            mel = F.pad(mel, (0, 3000 - T))
                        else:
                            mel = mel[:, :3000]

                        dual = mel.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1).contiguous()

                        out = self.model._compute_embedding(dual)
                        logit = out.view(-1)[0]
                        prob = torch.sigmoid(logit).item()

                        aug_logits.append(logit.item())
                        aug_probs.append(prob)

                    # aggregate per-window using *max* across augmentations (robust)
                    window_aug_probs.append(aug_probs)
                    window_aug_logits.append(aug_logits)

                    per_window_prob = float(max(aug_probs) if len(aug_probs) else 0.0)
                    per_window_logit = float(max(aug_logits) if len(aug_logits) else 0.0)

                    window_probs.append(per_window_prob)
                    window_logits.append(per_window_logit)

                if len(window_probs) == 0:
                    raise RuntimeError("No windows processed for audio file")

                mean_prob = float(np.mean(window_probs))

                # Determine top_k to aggregate logits
                num_windows = len(window_logits)
                if top_k is None or top_k <= 0:
                    k = max(1, int(max(1, np.ceil(top_pct * num_windows))))
                else:
                    k = min(max(1, int(top_k)), num_windows)

                # Sort logits descending and pick top-k
                sorted_logits = sorted(window_logits, reverse=True)
                topk_logits = sorted_logits[:k]
                agg_logit = float(np.mean(topk_logits))
                agg_prob = torch.sigmoid(torch.tensor(agg_logit)).item()

                # Consistency gate
                consistency_count = int(sum(1 for p in window_probs if p > consistency_threshold))
                consistency_pass = consistency_count >= consistency_min_count

                # Decision: require both agg_prob > 0.5 and consistency pass
                is_fake = (agg_prob > 0.5) and consistency_pass

                confidence = agg_prob if is_fake else (1 - agg_prob)

                # Top-k windows for output (by max per-window prob)
                max_per_window = [float(max(x) if len(x) else 0.0) for x in window_aug_probs]
                top_k_idxs = sorted(range(len(max_per_window)), key=lambda i: max_per_window[i], reverse=True)[:return_top_k]
                top_windows = [
                    {
                        "index": int(idx),
                        "start_sec": float(starts[idx] / sr),
                        "max_prob": float(max_per_window[idx]),
                        "mean_prob": float(window_probs[idx]),
                        "logit": float(window_logits[idx]),
                    }
                    for idx in top_k_idxs
                ]

                print("--- [LOGIT-AGG DIAGNOSTIC] ---")
                print(f"windows={num_windows} top_k={k} agg_prob={agg_prob:.4f} consistency={consistency_count}/{consistency_min_count}")

                return {
                    "is_fake": is_fake,
                    "confidence": confidence,
                    "raw_score": agg_prob,
                    "agg_logit": agg_logit,
                    "agg_prob": agg_prob,
                    "mean_score": mean_prob,
                    "window_probs": window_probs,
                    "window_logits": window_logits,
                    "window_aug_probs": window_aug_probs,
                    "top_windows": top_windows,
                    "consistency_count": consistency_count,
                    "consistency_pass": consistency_pass,
                    "topk_logits": topk_logits,
                }

        except Exception as e:
            print(f"Error: {e}")
            return {"is_fake": False, "confidence": 0, "raw_score": 0.5}

    def generate_calibration_csv(self, file_label_pairs, out_csv_path, **predict_kwargs):
        """Run inference on labeled files and save calibration features for Platt scaling.

        file_label_pairs: iterable of (path, label) where label is 0 (real) or 1 (fake)
        out_csv_path: output csv with columns: path,label,agg_logit,agg_prob,max_prob,mean_prob,topk_logits
        Additional kwargs are passed to `predict` (augmentation, normalization, etc.)
        """
        import csv

        rows = []
        for path, label in file_label_pairs:
            r = self.predict(path, **predict_kwargs)
            row = {
                "path": path,
                "label": int(label),
                "agg_logit": float(r.get("agg_logit", 0.0)),
                "agg_prob": float(r.get("agg_prob", 0.0)),
                "max_prob": float(max(r.get("window_probs", [0.0])) if r.get("window_probs") else 0.0),
                "mean_prob": float(r.get("mean_score", 0.0)),
                "topk_logits": ";".join([f"{v:.6f}" for v in r.get("topk_logits", [])]),
            }
            rows.append(row)

        # write csv
        with open(out_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "label", "agg_logit", "agg_prob", "max_prob", "mean_prob", "topk_logits"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        return out_csv_path