import logging
from pathlib import Path

import torch

from core.constants import Constants
from core.settings import DinoDetectorSettings, SamSegmenterSettings, DepthAnythingV2Settings
from image_utils import ImageUtils
from subject_pipeline import SubjectIsolationPipeline

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M')

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"DEVICE: {DEVICE}")

    dino_settings = DinoDetectorSettings(
        model_config_path="dino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="checkpoints/groundingdino_swint_ogc.pth",
        grounding_prompt=Constants.GROUNDING_PROMPT,
        box_threshold=0.3,
        text_threshold=0.25,
    )

    sam_settings = SamSegmenterSettings(
        checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
    )

    depth_settings = DepthAnythingV2Settings(
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        checkpoint_path="checkpoints/depth_anything_v2_vits.pth",
    )

    pipeline = SubjectIsolationPipeline(
        device=DEVICE,
        dino_settings=dino_settings,
        sam_segmenter_settings=sam_settings,
        depth_anything_settings=depth_settings,
    )

    run_folder = False
    if run_folder:
        folder_path = Path("photos")
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                subject_region, total_depth, dmin, dmax, image_source = pipeline.find_subject(file_path)
                ImageUtils.show_mask_depth(
                    mask=subject_region.mask,
                    image_source_depth=total_depth,
                    vmin=dmin,
                    vmax=dmax,
                    title=f"Current region"
                )
    else:
        file_path = Path("photos/good_light_1.png")
        subject_region, total_depth, dmin, dmax, image_source = pipeline.find_subject(file_path)


    def show_image(img, title=""):
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()


