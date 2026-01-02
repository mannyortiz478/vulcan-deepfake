# inference.py
import sys
import os
import torch
import librosa
import numpy as np
import yaml

# Ensure the src folder is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.models.whisper_meso_net import WhisperMesoNet

class MesoNetDetector:
    def __init__(self, model_pth_path, config_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        params = self.config['model']['parameters']
        params['device'] = self.device
        
        self.model = WhisperMesoNet(**params).to(self.device)
        
        checkpoint = torch.load(model_pth_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✅ WhisperMesoNet loaded on {self.device}")

    def predict(self, audio_path):
        """Processes audio and returns deepfake probability."""
        import torch.nn.functional as F
        import src.models.whisper_main as wm
        from src.models.whisper_main import N_SAMPLES, N_FFT, HOP_LENGTH
        
        # Helper to generate the Mel filters
        def get_mel_filters(device, n_mels=80):
            import librosa
            filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
            return torch.from_numpy(filters).to(device).float()

        with torch.no_grad():
            # 1. Load and Pad Audio
            audio, _ = librosa.load(audio_path, sr=16000)
            audio_tensor = torch.from_numpy(audio).to(self.device).float()
            
            if audio_tensor.shape[0] < N_SAMPLES:
                audio_tensor = F.pad(audio_tensor, (0, N_SAMPLES - audio_tensor.shape[0]))
            else:
                audio_tensor = audio_tensor[:N_SAMPLES]

            # 2. Manual Feature Extraction (Guaranteeing 201 bins)
            window = torch.hann_window(N_FFT).to(self.device)
            stft = torch.stft(
                audio_tensor, N_FFT, HOP_LENGTH, 
                window=window, center=True, return_complex=True
            )
            magnitudes = stft.abs() ** 2
            filters = get_mel_filters(self.device)
            
            # Match dimensions [80, 201] @ [201, 3001]
            mel = filters @ magnitudes
            mel = torch.log10(torch.clamp(mel, min=1e-10))

            # 3. Prepare for MesoNet (4D Tensor)
            # shape: [Batch, Channels, Height, Width] -> [1, 2, 80, 3001]
            mel_input = mel.unsqueeze(0).unsqueeze(0)
            dual_input = torch.cat([mel_input, mel_input], dim=1)

            # 4. BYPASS internal feature extraction
            # Instead of calling self.model(dual_input), we call the backend directly
            if hasattr(self.model, 'mesonet'):
                output = self.model.mesonet(dual_input)
            elif hasattr(self.model, 'classifier'):
                output = self.model.classifier(dual_input)
            else:
                # If we can't find the attribute, we call the feature-processing method
                # This depends on your src/models/whisper_meso_net.py structure
                output = self.model._compute_embedding(dual_input)
                if hasattr(self.model, 'fc'): # Final fully connected layer
                    output = self.model.fc(output)

            # 5. Result
            probability = torch.sigmoid(output).item()
            return {
                "is_fake": probability > 0.5,
                "confidence": probability if probability > 0.5 else (1 - probability),
                "raw_score": probability
            }