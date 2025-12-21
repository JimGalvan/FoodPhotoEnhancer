import cv2
import numpy as np


# ==================================================
# COLOR & LIGHTING
# ==================================================
def white_balance(img):
    """Gray-world white balance"""
    img = img.astype(np.float32)
    avg_b, avg_g, avg_r = img.mean(axis=(0, 1))
    avg = (avg_b + avg_g + avg_r) / 3
    img[:, :, 0] *= avg / avg_b
    img[:, :, 1] *= avg / avg_g
    img[:, :, 2] *= avg / avg_r
    return np.clip(img, 0, 255).astype(np.uint8)


def enhance_contrast(img):
    """CLAHE on luminance"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ==================================================
# FOOD COLOR BOOST
# ==================================================
def boost_saturation(img, amount=1.15):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= amount
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ==================================================
# TEXTURE SHARPENING
# ==================================================
def sharpen(img, strength=1.2):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img, strength, blur, -0.4, 0)


# ==================================================
# VIGNETTE
# ==================================================
def add_vignette(img, strength=0.25):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)

    mask = 1 - (dist / max_dist) * strength
    mask = np.clip(mask, 0.75, 1.0)

    out = img.astype(np.float32)
    out *= mask[:, :, np.newaxis]
    return out.astype(np.uint8)


# ==================================================
# MAIN ENHANCEMENT PIPELINE
# ==================================================
def enhance_food_plate(img_rgb):
    """
    img_rgb: RGB numpy array (plate crop, background already removed)
    """
    img = white_balance(img_rgb)
    img = enhance_contrast(img)
    img = boost_saturation(img, amount=1.2)
    img = sharpen(img, strength=1.25)
    img = add_vignette(img, strength=0.20)
    return img
