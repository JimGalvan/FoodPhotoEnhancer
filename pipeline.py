from core.settings import DinoDetectorSettings, SamSegmenterSettings, DepthAnythingV2Settings
from subject_pipeline import SubjectIsolationPipeline

if __name__ == "__main__":
    dino_settings = DinoDetectorSettings(
        model_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="checkpoints/groundingdino_swint_ogc.pth",
        grounding_prompt={"plate", "food"},
        box_threshold=0.3,
        text_threshold=0.25,
    )

    sam_settings = SamSegmenterSettings(
        checkpoint="checkpoints/sam_vit_b.pth",
        sam_model_type="vit_b",
    )

    depth_settings = DepthAnythingV2Settings(
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        checkpoint_path="checkpoints/depth_anything_v2_vits.pth",
    )

    pipeline = SubjectIsolationPipeline(
        device="cpu",
        dino_settings=dino_settings,
        sam_segmenter_settings=sam_settings,
        depth_anything_settings=depth_settings,
    )

