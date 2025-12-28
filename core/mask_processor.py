import cv2
import numpy as np


class MaskProcessor:
    @staticmethod
    def normalize(mask):
        mask = mask.astype(np.float32)
        return mask / max(mask.max(), 1e-6)

    @staticmethod
    def fill_holes(mask, kernel_size=7):
        mask_u8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        h, w = closed.shape
        flood = closed.copy()
        cv2.floodFill(flood, None, (0, 0), 255)

        filled = closed | cv2.bitwise_not(flood)
        return filled.astype(np.float32) / 255.0

    @staticmethod
    def soften(mask, sigma=3):
        return cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]
