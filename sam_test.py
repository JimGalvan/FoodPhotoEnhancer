import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------------------
# Helper: load image
# --------------------------------------------------
def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# --------------------------------------------------
# Config
# --------------------------------------------------
model_type = "vit_b"  # vit_b | vit_l | vit_h
checkpoint = "weights/sam_vit_b_01ec64.pth"
image_path = "photos/good_light_1.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load image
# --------------------------------------------------
image = load_image(image_path)

# --------------------------------------------------
# Load SAM
# --------------------------------------------------
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)
sam.eval()

# --------------------------------------------------
# Automatic mask generator
# --------------------------------------------------
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)

masks = mask_generator.generate(image)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
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

show_masks(masks, image)
