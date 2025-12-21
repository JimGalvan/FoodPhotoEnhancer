import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------------------
# Helper: load image
# --------------------------------------------------
def load_image_for_inference(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# --------------------------------------------------
# Configuration
# --------------------------------------------------
model_type = "vit_t"
sam_checkpoint = "weights/mobile_sam.pt"
image_path = "../photos/good_light_1.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load image
# --------------------------------------------------
image = load_image_for_inference(image_path)

# --------------------------------------------------
# Load MobileSAM
# --------------------------------------------------
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# --------------------------------------------------
# Automatic mask generation
# --------------------------------------------------
mask_generator = SamAutomaticMaskGenerator(
    model=mobile_sam,
    # points_per_side=32,          # higher = more detail, slower
    # pred_iou_thresh=0.86,
    # stability_score_thresh=0.92,
    # crop_n_layers=1,
    # crop_n_points_downscale_factor=2,
    # min_mask_region_area=100     # remove small specks
)

masks = mask_generator.generate(image)

# --------------------------------------------------
# Visualization helpers
# --------------------------------------------------
def show_automatic_masks(masks, image):
    if len(masks) == 0:
        return

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

# --------------------------------------------------
# Display result
# --------------------------------------------------
show_automatic_masks(masks, image)
