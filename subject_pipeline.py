import logging

import numpy as np

from core.depth import Depth
from core.models import RegionPrompt
from dino_detector import DinoDetector
from image_utils import ImageUtils
from loaders import load_grounding_dino, load_sam, load_depth_anything_v2
from sam_segmenter import SamSegmenter


class DinoDetectorSettings:
    def __init__(
            self,
            model_config_path: str,
            model_checkpoint_path: str,
            grounding_prompt: any,
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
            checkpoint_path: str,
            sam_model_type: str
    ):
        self.sam_model_type = sam_model_type
        self.checkpoint = checkpoint_path


class DepthAnythingV2Settings:
    def __init__(
            self,
            encoder="vits",
            features=64,
            out_channels=(48, 96, 192, 384),
            checkpoint_path="checkpoints/depth_anything_v2_vits.pth",
            device="cpu",
    ):
        self.encoder = encoder
        self.features = features
        self.out_channels = list(out_channels)
        self.checkpoint_path = checkpoint_path
        self.device = device


class SubjectIsolationPipeline:
    logger = logging.getLogger(__name__)

    def __init__(
            self,
            device: str,
            dino_settings: DinoDetectorSettings,
            sam_segmenter_settings: SamSegmenterSettings,
            depth_anything_settings: DepthAnythingV2Settings
    ):
        self.logger.info("Initializing SubjectIsolationPipeline")
        print(f"SubjectIsolationPipeline device: {device}")
        if device not in ["cpu", "cuda"]:
            self.logger.error(f"Unsupported device: {device}")
            raise ValueError(f"Device {device} is not supported")

        self.device = device
        self.dino_settings = dino_settings
        self.sam_segmenter_settings = sam_segmenter_settings
        self.depth_anything_settings = depth_anything_settings

        # Load DINO
        self.logger.debug("Loading Grounding DINO model")
        dino_model = load_grounding_dino(
            settings=self.dino_settings
        )
        self.dino_detector = DinoDetector(
            device=self.device,
            model=dino_model
        )

        # Load SAM
        self.logger.debug("Loading SAM predictor")
        self.sam_predictor = load_sam(
            self.sam_segmenter_settings,
            device=self.device
        )

        # Load Depth Anything
        self.logger.debug("Loading Depth Anything v2 model")
        self.depth_anything_model = load_depth_anything_v2(
            settings=self.depth_anything_settings,
            device=self.device
        )

        self.logger.info("SubjectIsolationPipeline initialized successfully")

    def find_subject(self, image_path):
        self.logger.info("Starting subject detection pipeline")

        # 1. Detect items (DINO)
        self.logger.debug("Running DINO box detection")
        image_source, image, boxes, phrases = self.dino_detector.detect_boxes(
            image_path,
            grounding_prompt=self.dino_settings.grounding_prompt,
            box_threshold=self.dino_settings.box_threshold,
            text_threshold=self.dino_settings.text_threshold,
        )

        regions: list[RegionPrompt] = []
        for box, phrase in zip(boxes, phrases):
            regions.append(RegionPrompt(box=box, label=phrase))

        self.logger.debug(f"Detected {len(regions)} regions")

        pil_img = ImageUtils.convert_img_source_to_pil(image_source)

        # 2. Segment items (SAM)
        self.logger.debug("Running SAM segmentation")
        sam_segmenter = SamSegmenter(
            predictor=self.sam_predictor,
            device=self.device
        )
        regions_with_masks = sam_segmenter.segment(
            image=ImageUtils.pil_to_numpy(pil_img),
            regions=regions
        )

        # 3. Compute total Depth from image
        self.logger.debug("Computing depth map")
        total_depth = self.depth_anything_model.infer_image(
            raw_image=image_source,
            input_size=618
        )
        dmin = np.nanmin(total_depth)
        dmax = np.nanmax(total_depth)

        self.logger.debug(f"Depth range: min={dmin:.4f}, max={dmax:.4f}")

        # 4. Determine main dish
        self.logger.debug("Determining main dish")
        main_dish = Depth.compute_main_dish(
            total_depth,
            regions_with_masks,
            image_source
        )

        # 5. Determine items related to dish
        self.logger.debug("Finding subject region")
        subject_region = Depth.find_related_dish_region(
            depth=total_depth,
            main_dish=main_dish,
            regions_with_masks=regions_with_masks,
            dmin=dmin,
            dmax=dmax,
            max_attempts=5
        )

        self.logger.info("Subject detection pipeline completed")

        return subject_region, total_depth, dmin, dmax, image_source
