class DinoDetectorSettings:
    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        grounding_prompt: set,
        box_threshold: float,
        text_threshold: float,
    ):
        self.model_config_path = model_config_path
        self.model_checkpoint_path = model_checkpoint_path
        self.grounding_prompt = grounding_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold


class SamSegmenterSettings:
    def __init__(
        self,
        checkpoint="checkpoints/sam_vit_b.pth",
        sam_model_type="vit_b",
    ):
        self.checkpoint = checkpoint
        self.sam_model_type = sam_model_type


class DepthAnythingV2Settings:
    def __init__(
        self,
        encoder: str = "vits",
        features: int = 64,
        out_channels=(48, 96, 192, 384),
        checkpoint_path: str = "checkpoints/depth_anything_v2_vits.pth",
    ):
        self.encoder = encoder
        self.features = features
        self.out_channels = list(out_channels)
        self.checkpoint_path = checkpoint_path
