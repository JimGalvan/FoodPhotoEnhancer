import cv2
import numpy as np

from image_utils import ImageUtils


class LightingUtils:
    @staticmethod
    def window_light(rgb, gamma):
        img_u8 = ImageUtils.to_uint8(rgb)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(lab)
        L = np.power(L.astype(np.float32) / 255.0, gamma)
        L = ImageUtils.to_uint8(L)

        out = cv2.merge([L, A, B])
        return ImageUtils.to_float01(cv2.cvtColor(out, cv2.COLOR_LAB2RGB))

    @staticmethod
    def clahe(rgb, clip_limit, strength):
        img_u8 = ImageUtils.to_uint8(rgb)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        Lc = clahe.apply(L)

        Lout = cv2.addWeighted(L, 1.0 - strength, Lc, strength, 0)
        out = cv2.merge([Lout, A, B])

        return ImageUtils.to_float01(cv2.cvtColor(out, cv2.COLOR_LAB2RGB))
