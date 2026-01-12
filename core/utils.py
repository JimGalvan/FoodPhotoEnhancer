import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_masks(masks, image):
    overlay = image.copy()

    for m in masks:
        color = np.random.randint(0, 255, size=3)
        overlay[m["segmentation"]] = (
                0.5 * overlay[m["segmentation"]] + 0.5 * color
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay.astype(np.uint8))
    plt.axis("off")
    plt.show()


def clahe_gray(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def histogram_equalize_rgb(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def gray_to_sam_input(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_prompt_masks(masks, image, alpha=0.5, input_box = None):
    overlay = image.copy()

    # Ensure masks is (N, H, W)
    if masks.ndim == 2:
        masks = masks[None, ...]

    for mask in masks:
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        overlay[mask] = (
                (1 - alpha) * overlay[mask] + alpha * color
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay.astype(np.uint8))
    if input_box.any():
        show_box(input_box, plt.gca())
    plt.axis("off")
    plt.show()


def get_prompt_box(image, s = 1):
    height, width = image.shape[:2]

    # Box dimensions
    box_w = s * width
    box_h = s * height

    # Image center
    cx = width / 2
    cy = height / 2

    # Centered box coordinates
    A = int(cx - box_w / 2)
    B = int(cy - box_h / 2)
    C = int(cx + box_w / 2)
    D = int(cy + box_h / 2)

    return np.array([A, B, C, D])



