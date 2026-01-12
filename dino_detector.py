import logging

from groundingdino.util.inference import predict, load_image
from pycparser.ply.yacc import resultlimit

from core.models import Box
from image_utils import ImageUtils


class DinoDetector:
    logger = logging.getLogger(__name__)

    def __init__(self, device, model):
        self.device = device
        self.model = model

    def detect_boxes(self, image_path: str, grounding_prompt, box_threshold=0.25, text_threshold=0.25):
        self.logger.info(f"Image path: {image_path}")
        self.logger.info(f"Grounding prompt: {grounding_prompt}")
        self.logger.info("Loading image for DINO box detection...")
        ImageUtils.wait_for_image(
            image_path=image_path,
            timeout=10,
        )

        image_source, image = load_image(image_path)
        self.logger.info(f"Image source: {image_source}")

        h, w, _ = image_source.shape

        self.logger.info("Running DINO box detection...")
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=grounding_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        
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
        return image_source, image, result_boxes, phrases
