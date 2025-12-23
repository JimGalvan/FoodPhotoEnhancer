import threading
import numpy as np
import torch
from typing import Dict, Tuple
from segment_anything import SamPredictor


class SamSegmenter:
    def __init__(
        self,
        predictor: SamPredictor,
        max_cache_size: int = 8,
        device: str = "cuda",
    ):
        self.predictor = predictor
        self.device = device

        self._lock = threading.Lock()
        self._embedding_cache: Dict[int, torch.Tensor] = {}
        self._cache_order = []
        self._max_cache_size = max_cache_size

    def _image_key(self, image: np.ndarray) -> int:
        """Fast hash key for caching."""
        return hash(image.tobytes())

    @torch.inference_mode()
    def _set_image_cached(self, image: np.ndarray):
        key = self._image_key(image)

        if key in self._embedding_cache:
            self.predictor.features = self._embedding_cache[key]
            return

        self.predictor.set_image(image)

        features = self.predictor.features
        self._embedding_cache[key] = features
        self._cache_order.append(key)

        # LRU eviction
        if len(self._cache_order) > self._max_cache_size:
            old_key = self._cache_order.pop(0)
            del self._embedding_cache[old_key]
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def segment(self, image: np.ndarray, boxes):
        with self._lock:
            self._set_image_cached(image)

            masks = []
            for box in boxes:
                box = np.asarray(box, dtype=np.float32)
                mask, _, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False,
                )
                masks.append(mask[0].astype(bool))

            return masks
