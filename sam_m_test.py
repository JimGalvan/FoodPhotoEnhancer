import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor

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
image_path = "photos/good_light_1.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load image
# --------------------------------------------------
img = load_image_for_inference(image_path)

# --------------------------------------------------
# Load MobileSAM
# --------------------------------------------------
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)
predictor.set_image(img)

# --------------------------------------------------
# Define prompt (REQUIRED)
# Example: one foreground click
# --------------------------------------------------
h, w, _ = img.shape
point_coords = np.array([[w // 2, h // 2]])  # center point
point_labels = np.array([1])                 # 1 = foreground

# --------------------------------------------------
# Run prediction
# --------------------------------------------------
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)

# --------------------------------------------------
# Select best mask
# --------------------------------------------------
best_mask = masks[np.argmax(scores)]

# --------------------------------------------------
# Display result
# --------------------------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(best_mask, alpha=0.5)
plt.scatter(point_coords[:, 0], point_coords[:, 1], c="lime", s=100, marker="*")
plt.axis("off")
plt.show()
