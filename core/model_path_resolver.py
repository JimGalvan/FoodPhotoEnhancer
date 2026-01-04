import os


class ModelPathResolver:
    DEFAULT_PATHS = {
        "dino_checkpoint": "checkpoints/groundingdino_swint_ogc.pth",
        "sam_checkpoint": "checkpoints/sam_vit_b_01ec64.pth",
        "depth_checkpoint": "checkpoints/depth_anything_v2_vits.pth",
        "dino_config": "dino/config/GroundingDINO_SwinT_OGC.py",
    }

    @classmethod
    def get_dino_checkpoint(cls):
        return os.getenv("DINO_CHECKPOINT_URL", cls.DEFAULT_PATHS["dino_checkpoint"])

    @classmethod
    def get_sam_checkpoint(cls):
        return os.getenv("SAM_CHECKPOINT_URL", cls.DEFAULT_PATHS["sam_checkpoint"])

    @classmethod
    def get_depth_checkpoint(cls):
        return os.getenv("DEPTH_CHECKPOINT_URL", cls.DEFAULT_PATHS["depth_checkpoint"])

    @classmethod
    def get_dino_config(cls):
        return cls.DEFAULT_PATHS["dino_config"]
