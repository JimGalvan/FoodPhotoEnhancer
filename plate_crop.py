import os
import torch
import numpy as np
import open_clip
from PIL import Image

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

PLATE_CROP_DIR = "plate_crops"
os.makedirs(PLATE_CROP_DIR, exist_ok=True)


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
        "models/groundingdino_swint_ogc.pth",
    )


def load_sam(checkpoint_path, model_type="vit_b"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


# ==================================================
# CLIP CLASSIFICATION
# ==================================================
def classify_clip(model, preprocess, tokenizer, crop):
    texts = FOOD_TEXTS + CONTAINER_TEXTS + ["not food"]

    image = preprocess(crop).unsqueeze(0).to(DEVICE)
    text = tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        img_f = model.encode_image(image)
        txt_f = model.encode_text(text)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)
        probs = (100 * img_f @ txt_f.T).softmax(dim=-1)

    idx = probs.argmax().item()
    score = probs[0, idx].item()
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
def dino_detect_boxes(image_path, dino_model):
    image_bgr, image = load_image(image_path)
    h, w, _ = image_bgr.shape

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
        if x2 > x1 and y2 > y1:
            xyxy.append((x1, y1, x2, y2))

    return image_bgr, xyxy


# ==================================================
# SAM SEGMENTATION
# ==================================================
def sam_segment(predictor, image_bgr, boxes):
    image_rgb = image_bgr[:, :, ::-1]
    predictor.set_image(image_rgb)

    masks = []
    for box in boxes:
        mask, _, _ = predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
        )
        masks.append(mask[0].astype(bool))
    return masks


# ==================================================
# MASK UTILITIES
# ==================================================
def mask_overlap_ratio(inner, outer):
    return np.logical_and(inner, outer).sum() / max(inner.sum(), 1)


# ==================================================
# MAIN PIPELINE
# ==================================================
def generate_plate_crops(
    image_path,
    sam_checkpoint,
    sam_model_type="vit_b",
):
    clip_model, preprocess, tokenizer = load_openclip()
    dino_model = load_grounding_dino()
    sam_predictor = load_sam(sam_checkpoint, sam_model_type)

    image_bgr, boxes = dino_detect_boxes(image_path, dino_model)
    image_rgb = image_bgr[:, :, ::-1]
    pil_img = Image.fromarray(image_rgb)

    labels = []
    kept_boxes = []

    for box in boxes:
        label = classify_clip(
            clip_model, preprocess, tokenizer, pil_img.crop(box)
        )
        if label:
            labels.append(label)
            kept_boxes.append(box)

    if not kept_boxes:
        print("No plates found.")
        return

    masks = sam_segment(sam_predictor, image_bgr, kept_boxes)

    food_masks = []
    plate_masks = []

    for label, mask in zip(labels, masks):
        if label == "food":
            food_masks.append(mask)
        elif label == "plate":
            plate_masks.append(mask)

    plate_idx = 0

    for plate_mask in plate_masks:
        food_on_plate = [
            fm for fm in food_masks
            if mask_overlap_ratio(fm, plate_mask) >= CONTAINMENT_THRESHOLD
        ]

        if not food_on_plate:
            continue

        # ---- crop region = union of plate + food masks ----
        union_mask = plate_mask.copy()
        for fm in food_on_plate:
            union_mask |= fm

        ys, xs = np.where(union_mask)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # IMPORTANT: do NOT mask pixels out
        crop = image_rgb[y1:y2, x1:x2]

        out_path = os.path.join(
            PLATE_CROP_DIR, f"plate_{plate_idx}.png"
        )
        Image.fromarray(crop).save(out_path)
        plate_idx += 1

    print(f"Saved {plate_idx} plate crops (with full food) to '{PLATE_CROP_DIR}/'")



# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    generate_plate_crops(
        image_path="photos/wood_dish_1.png",
        sam_checkpoint="models/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
    )
