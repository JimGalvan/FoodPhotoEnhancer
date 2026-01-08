import logging

import numpy as np

from core.mask_processor import MaskProcessor
from core.photo_enhancer import PhotoEnhancer
from image_utils import ImageUtils

logger = logging.getLogger(__name__)


class EnhancementPipeline:

    def __init__(self, image, mask):
        self.image = ImageUtils.to_float01(image)
        self.mask = MaskProcessor.normalize(mask)

    def run(self):
        logger.info("pipeline:start")

        logger.info("mask:fill_holes")
        mask_filled = MaskProcessor.fill_holes(self.mask, kernel_size=16)
        logger.debug(
            "mask_filled stats | min=%.3f max=%.3f mean=%.3f",
            mask_filled.min(),
            mask_filled.max(),
            mask_filled.mean(),
        )

        logger.info("mask:soften")
        mask_soft = MaskProcessor.soften(mask_filled, sigma=0.005)
        logger.debug("mask_soft shape=%s", mask_soft.shape)

        logger.info("subject:enhance")
        subject = PhotoEnhancer.enhance(self.image)
        logger.debug(
            "subject stats | min=%.3f max=%.3f mean=%.3f",
            subject.min(),
            subject.max(),
            subject.mean(),
        )

        logger.info("subject:food_color_boost")
        subject = PhotoEnhancer.boost_food_colors(subject)

        logger.info("background:blur")
        background = ImageUtils.gaussian_blur(self.image, sigma=3)

        logger.info("background:vignette")
        vignette = PhotoEnhancer.create_vignette(background.shape)
        background = background * vignette

        logger.info("composite:blend")
        final = mask_soft * subject + (1.0 - mask_soft) * background
        logger.debug(
            "final stats | min=%.3f max=%.3f mean=%.3f",
            final.min(),
            final.max(),
            final.mean(),
        )

        logger.info("pipeline:done")
        return final, mask_filled
