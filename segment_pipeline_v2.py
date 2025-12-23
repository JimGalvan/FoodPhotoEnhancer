import os
import torch
import numpy as np
import open_clip
from PIL import Image, ImageDraw

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


# ==================================================
# CONFIG
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_PROMPT = (
    "food, meal, cooked food, dish, plate, bowl, wooden plate, ceramic plate"
)

FOOD_TEXTS = ["food", "meal", "cooked food"]
CONTAINER_TEXTS = ["plate", "bowl", "dish", "wooden plate", "ceramic plate"]

CLIP_THRESHOLD = 0.30
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
CONTAINMENT_THRESHOLD = 0.30  # % of food mask inside plate mask


# ==================================================
# MODEL LOADERS
# ==================================================
def load_openclip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def load_grounding_dino():
    return load_model(
        "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "checkpoints/groundingdino_swint_ogc.pth",
    )


def load_sam(checkpoint_path: str, model_type: str = "vit_b") -> SamPredictor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


# ==================================================
# CLIP CLASSIFICATION
# ==================================================
def classify_clip(model, preprocess, tokenizer, crop: Image.Image):
    texts = FOOD_TEXTS + CONTAINER_TEXTS + ["not food"]

    image = preprocess(crop).unsqueeze(0).to(DEVICE)
    text = tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        img_f = model.encode_image(image)
        txt_f = model.encode_text(text)

        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)

        probs = (100 * img_f @ txt_f.T).softmax(dim=-1)

    idx = int(probs.argmax().item())
    score = float(probs[0, idx].item())
    label = texts[idx]

    if score < CLIP_THRESHOLD:
        return None
    if label in FOOD_TEXTS:
        return "food"
    if label in CONTAINER_TEXTS:
        return "plate"
    return None


# ==================================================
# DETECTION (GroundingDINO)
# ==================================================
def dino_detect_boxes(image_path: str, dino_model):
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    boxes, _, _ = predict(
        model=dino_model,
        image=image,
        caption=GROUNDING_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    xyxy = []
    for b in boxes:
        cx, cy, bw, bh = b.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 > x1 and y2 > y1:
            xyxy.append((x1, y1, x2, y2))

    return image_source, xyxy


# ==================================================
# SEGMENTATION (SAM)
# ==================================================
def sam_segment(predictor: SamPredictor, image_bgr, boxes):
    image_rgb = image_bgr[:, :, ::-1]
    predictor.set_image(image_rgb)

    masks = []
    for box in boxes:
        box = np.array(box, dtype=np.float32)
        mask, _, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )
        masks.append(mask[0].astype(bool))

    return masks


# ==================================================
# MASK CONTAINMENT
# ==================================================
def mask_overlap_ratio(inner_mask, outer_mask):
    intersection = np.logical_and(inner_mask, outer_mask).sum()
    inner_area = inner_mask.sum()
    if inner_area == 0:
        return 0.0
    return intersection / inner_area


# ==================================================
# VISUALIZATION
# ==================================================
def overlay_mask(image, mask, color, alpha):
    out = image.copy().astype(np.float32)
    color = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color
    return out.astype(np.uint8)


# ==================================================
# MAIN PIPELINE
# ==================================================
def grounded_sam_food_containment(
    image_path: str,
    output_path: str,
    sam_checkpoint: str,
    sam_model_type: str = "vit_b",
):
    clip_model, preprocess, tokenizer = load_openclip()
    dino_model = load_grounding_dino()
    sam_predictor = load_sam(sam_checkpoint, sam_model_type)

    image_bgr, boxes = dino_detect_boxes(image_path, dino_model)
    pil_img = Image.fromarray(image_bgr[:, :, ::-1])

    kept = []
    kept_boxes = []

    for box in boxes:
        crop = pil_img.crop(box)
        label = classify_clip(clip_model, preprocess, tokenizer, crop)
        if label:
            kept.append({"box": box, "label": label})
            kept_boxes.append(box)

    if not kept:
        pil_img.save(output_path)
        return

    masks = sam_segment(sam_predictor, image_bgr, kept_boxes)

    foods, plates = [], []
    for item, mask in zip(kept, masks):
        if item["label"] == "food":
            foods.append({"box": item["box"], "mask": mask})
        else:
            plates.append({"box": item["box"], "mask": mask})

    rgb = image_bgr[:, :, ::-1]

    for plate in plates:
        contained_foods = []
        for food in foods:
            overlap = mask_overlap_ratio(food["mask"], plate["mask"])
            if overlap >= CONTAINMENT_THRESHOLD:
                contained_foods.append(food)

        if not contained_foods:
            continue

        rgb = overlay_mask(rgb, plate["mask"], (0, 120, 255), 0.35)

        for food in contained_foods:
            rgb = overlay_mask(rgb, food["mask"], (255, 0, 0), 0.45)

    Image.fromarray(rgb).save(output_path)
    print("Saved:", output_path)


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    grounded_sam_food_containment(
        image_path="photos/wood_dish_1.png",
        output_path="test_output/wood_dish_1_mask.png",
        sam_checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
    )
