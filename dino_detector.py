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

        # Warmup the model to prevent first-run hangs in remote GPU environments
        self._warmup_model()

    def _warmup_model(self):
        """Warmup model with a dummy inference to initialize all components properly."""
        try:
            self.logger.info("Warming up DINO model...")

            # Create a small dummy image (224x224 RGB)
            dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 128

            # Preprocess with explicit device handling
            self.logger.info("Preprocessing dummy image...")
            dummy_tensor = Model.preprocess_image(dummy_image)

            if isinstance(dummy_tensor, torch.Tensor):
                dummy_tensor = dummy_tensor.to(self.device)

            # Sync if CUDA
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()

            # Run a dummy prediction with a simple prompt
            self.logger.info("Running dummy prediction...")
            with torch.no_grad():
                _ = predict(
                    model=self.model,
                    image=dummy_tensor,
                    caption="object",
                    box_threshold=0.25,
                    text_threshold=0.25,
                    device=self.device,
                )

            # Final sync
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            self.logger.info("DINO model warmup complete")
        except Exception as e:
            self.logger.warning(f"Model warmup failed (non-critical): {e}")

    def detect_boxes(self, image_bgr: np.ndarray, grounding_prompt, box_threshold=0.25, text_threshold=0.25):
        self.logger.info(f"Grounding prompt: {grounding_prompt}")
        self.logger.info("Loading image for DINO box detection...")

        # Preprocess image
        self.logger.info("Step 1: Preprocessing image...")
        image_tensor = Model.preprocess_image(image_bgr)
        self.logger.info(f"Image tensor shape: {image_tensor.shape if hasattr(image_tensor, 'shape') else 'unknown'}")

        # Explicitly move tensor to the correct device
        self.logger.info(f"Step 2: Moving tensor to device {self.device}...")
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.to(self.device)
        self.logger.info("Tensor moved to device")

        h, w, _ = image_bgr.shape

        # Clear CUDA cache before inference (helps with remote GPU environments)
        if self.device.startswith('cuda') and torch.cuda.is_available():
            self.logger.info("Step 3: Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("CUDA cache cleared and synchronized")

        self.logger.info("Step 4: Running DINO prediction...")

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=grounding_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        self.logger.info(f"Predict returned {len(boxes) if boxes is not None else 0} boxes")

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
