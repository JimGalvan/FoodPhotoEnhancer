import os
import torch

from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor

from core.settings import DinoDetectorSettings, SamSegmenterSettings, DepthAnythingV2Settings
from depth_anything_v2.dpt import DepthAnythingV2


def load_grounding_dino(settings: DinoDetectorSettings):
    return load_model(
        settings.model_config_path,
        settings.model_checkpoint_path,
    )


def load_sam(
        settings: SamSegmenterSettings,
        device: str = "cpu",
) -> SamPredictor:
    if not os.path.exists(settings.checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {settings.checkpoint}"
        )

    sam = sam_model_registry[settings.sam_model_type](
        checkpoint=settings.checkpoint
    )
    sam.to(device=device)
    return SamPredictor(sam)


def load_depth_anything_v2(
        settings: DepthAnythingV2Settings,
        device: str = "cpu",
):
    model = DepthAnythingV2(
        encoder=settings.encoder,
        features=settings.features,
        out_channels=settings.out_channels,
    )

    state = torch.load(
        settings.checkpoint_path,
        map_location=device
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
