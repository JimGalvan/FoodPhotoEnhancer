import numpy as np

from core.lighting_utils import LightingUtils


class SubjectEnhancer:
    @staticmethod
    def enhance(image):
        x = image.copy()
        x *= 1.15
        x[..., 0] *= 1.03
        x[..., 2] *= 0.97
        x = np.clip(x, 0.0, 1.0)

        x = LightingUtils.window_light(x, gamma=0.90)

        mean = x.mean(axis=(0, 1), keepdims=True)
        x = x + 0.05 * (x - mean)

        gray = np.mean(x, axis=2, keepdims=True)
        x = gray + 1.04 * (x - gray)

        x = np.clip(x, 0.0, 1.0)
        x = LightingUtils.clahe(x, clip_limit=3.5, strength=0.4)

        return x
