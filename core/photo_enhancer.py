import numpy as np

from core.lighting_utils import LightingUtils


class PhotoEnhancer:
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

    @staticmethod
    def boost_food_colors(image, saturation=0.90, warmth_red=1.05, warmth_yellow=1.02):
        gray = image.mean(axis=2, keepdims=True)
        enhanced = gray + saturation * (image - gray)
        enhanced[..., 0] *= warmth_red
        enhanced[..., 1] *= warmth_yellow
        return np.clip(enhanced, 0.0, 1.0)

    @staticmethod
    def desaturate_background(image, amount=0.3):
        gray = image.mean(axis=2, keepdims=True)
        return gray + amount * (image - gray)

    @staticmethod
    def create_vignette(shape, strength=0.3):
        h, w = shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1.0 - strength * (radius / max_radius) ** 2
        return vignette[..., np.newaxis]
