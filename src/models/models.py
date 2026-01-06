from typing import Dict


def get_model(model_name: str, config: Dict, device: str):
    # lazy imports to avoid importing heavy optional deps at module import time
    if model_name == "rawnet3":
        from src.models import rawnet3

        return rawnet3.prepare_model()
    elif model_name == "lcnn":
        from src.models import lcnn

        return lcnn.FrontendLCNN(device=device, **config)
    elif model_name == "specrnet":
        from src.models import specrnet

        return specrnet.FrontendSpecRNet(
            device=device,
            **config,
        )
    elif model_name == "mesonet":
        from src.models import meso_net

        return meso_net.FrontendMesoInception4(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_lcnn":
        from src.models import whisper_lcnn

        return whisper_lcnn.WhisperLCNN(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "whisper_specrnet":
        from src.models import whisper_specrnet

        return whisper_specrnet.WhisperSpecRNet(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "whisper_mesonet":
        from src.models import whisper_meso_net

        return whisper_meso_net.WhisperMesoNet(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", True),
            fc1_dim=config.get("fc1_dim", 1024),
            device=device,
        )
    elif model_name == "whisper_frontend_lcnn":
        from src.models import whisper_lcnn

        return whisper_lcnn.WhisperMultiFrontLCNN(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_specrnet":
        from src.models import whisper_specrnet

        return whisper_specrnet.WhisperMultiFrontSpecRNet(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_mesonet":
        from src.models import whisper_meso_net

        return whisper_meso_net.WhisperMultiFrontMesoNet(
            input_channels=config.get("input_channels", 2),
            fc1_dim=config.get("fc1_dim", 1024),
            freeze_encoder=config.get("freeze_encoder", True),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported")
