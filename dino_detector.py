import logging

import numpy as np
import torch
from groundingdino.util.inference import predict, Model
from pycparser.ply.yacc import resultlimit

from core.models import Box
from image_utils import ImageUtils


class DinoDetector:
    logger = logging.getLogger(__name__)

    def __init__(self, device, model):
        self.device = device
        self.model = model

    def detect_boxes(self, image_bgr: np.ndarray, grounding_prompt, box_threshold=0.25, text_threshold=0.25):
        self.logger.info(f"Grounding prompt: {grounding_prompt}")
        self.logger.info("Loading image for DINO box detection...")
        image_tensor = Model.preprocess_image(image_bgr)

        # Explicitly move tensor to the correct device
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.to(self.device)

        h, w, _ = image_bgr.shape

        # Clear CUDA cache before inference (helps with remote GPU environments)
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.logger.info("Running DINO box detection...")
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=grounding_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # Synchronize after inference to ensure completion
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()

        self.logger.info("DINO box detection complete")
        result_boxes: list[Box] = []
        for b in boxes:
            cx, cy, bw, bh = b.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                box = Box(x1, y1, x2, y2)
                result_boxes.append(box)
        return image_bgr, image_tensor, result_boxes, phrases
