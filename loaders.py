import os
import torch

from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor

from depth_anything.depth_anything_v2.dpt import DepthAnythingV2
from core.settings import DinoDetectorSettings, SamSegmenterSettings, DepthAnythingV2Settings
from core.model_downloader import ModelDownloader


def load_grounding_dino(settings: DinoDetectorSettings):
    # Download checkpoint if it's a URL
    checkpoint_path = ModelDownloader.get_local_path(settings.model_checkpoint_path)

    return load_model(
        settings.model_config_path,
        checkpoint_path,
    )


def load_sam(
        settings: SamSegmenterSettings,
        device: str = "cpu",
) -> SamPredictor:
    # Download checkpoint if it's a URL
    checkpoint_path = ModelDownloader.get_local_path(settings.checkpoint)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}"
        )

    sam = sam_model_registry[settings.sam_model_type](
        checkpoint=checkpoint_path
    )
    sam.to(device=device)
    return SamPredictor(sam)


def load_depth_anything_v2(
        settings: DepthAnythingV2Settings,
        device: str = "cpu",
):
    # Download checkpoint if it's a URL
    checkpoint_path = ModelDownloader.get_local_path(settings.checkpoint_path)

    model = DepthAnythingV2(
        encoder=settings.encoder,
        features=settings.features,
        out_channels=settings.out_channels,
    )

    state = torch.load(
        checkpoint_path,
        map_location=device
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
