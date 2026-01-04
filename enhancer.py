from functools import lru_cache
import logging
import torch

from core.constants import Constants
from core.model_path_resolver import ModelPathResolver
from core.settings import (
    DinoDetectorSettings,
    SamSegmenterSettings,
    DepthAnythingV2Settings,
)
from subject_pipeline import SubjectIsolationPipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_subject_isolation_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Subject seg pipeline device: %s", device)

    dino_settings = DinoDetectorSettings(
        model_config_path=ModelPathResolver.get_dino_config(),
        model_checkpoint_path=ModelPathResolver.get_dino_checkpoint(),
        grounding_prompt=Constants.GROUNDING_PROMPT,
        box_threshold=0.30,
        text_threshold=0.30,
    )

    sam_settings = SamSegmenterSettings(
        checkpoint=ModelPathResolver.get_sam_checkpoint(),
        sam_model_type="vit_b",
    )

    depth_settings = DepthAnythingV2Settings(
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        checkpoint_path=ModelPathResolver.get_depth_checkpoint(),
    )

    return SubjectIsolationPipeline(
        device=device.type,
        dino_settings=dino_settings,
        sam_segmenter_settings=sam_settings,
        depth_anything_settings=depth_settings,
    )
