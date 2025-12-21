import os
import torch
import numpy as np
import open_clip
from PIL import Image, ImageDraw

from groundingdino.util.inference import load_model, load_image, predict

from segment_anything import sam_model_registry, SamPredictor


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_PROMPT = (
    "food, meal, cooked food, dish, plate, bowl, wooden plate, ceramic plate, food bowl"
)

FOOD_TEXTS = ["food", "meal", "cooked food"]
CONTAINER_TEXTS = ["plate", "bowl", "dish", "wooden plate", "ceramic plate"]

CLIP_THRESHOLD = 0.30

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25


# -----------------------------
# LOAD MODELS
# -----------------------------
def load_openclip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def load_grounding_dino():
    return load_model(
        "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "models/groundingdino_swint_ogc.pth",
    )

def load_sam(checkpoint_path: str, model_type: str = "vit_b") -> SamPredictor:
    """
    model_type: 'vit_b', 'vit_l', or 'vit_h'
    checkpoint_path: e.g. 'sam_vit_b_01ec64.pth'
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}\n"
            f"Download it (example):\n"
            f"  curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


# -----------------------------
# CLIP FILTERING
# -----------------------------
def classify_clip(clip_model, clip_preprocess, clip_tokenizer, crop_pil: Image.Image):
    texts = FOOD_TEXTS + CONTAINER_TEXTS + ["not food"]

    image = clip_preprocess(crop_pil).unsqueeze(0).to(DEVICE)
    text = clip_tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        img_f = clip_model.encode_image(image)
        txt_f = clip_model.encode_text(text)

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


# -----------------------------
# DETECTION (DINO)
# -----------------------------
def dino_detect_boxes(image_path: str, dino_model):
    image_source, image = load_image(image_path)  # image_source is BGR np array
    h, w, _ = image_source.shape

    boxes, _, _ = predict(
        model=dino_model,
        image=image,
        caption=GROUNDING_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    # Convert cx,cy,w,h (normalized) -> xyxy pixel ints
    xyxy = []
    for b in boxes:
        cx, cy, bw, bh = b.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            xyxy.append((x1, y1, x2, y2))

    return image_source, xyxy  # image_source is BGR uint8


# -----------------------------
# SEGMENTATION (SAM)
# -----------------------------
def sam_segment(predictor: SamPredictor, image_bgr: np.ndarray, boxes_xyxy):
    """
    Returns list of masks (H,W) bool arrays aligned to image.
    """
    # SAM expects RGB
    image_rgb = image_bgr[:, :, ::-1]
    predictor.set_image(image_rgb)

    masks = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        mask, _, _ = predictor.predict(
            box=box,
            point_coords=None,
            point_labels=None,
            multimask_output=False,
        )
        masks.append(mask[0].astype(bool))
    return masks


# -----------------------------
# VISUALIZATION
# -----------------------------
def overlay_mask_rgba(base_rgb: np.ndarray, mask: np.ndarray, color_rgb, alpha: float = 0.45):
    """
    base_rgb: HxWx3 uint8 RGB
    mask: HxW bool
    color_rgb: tuple (r,g,b)
    """
    out = base_rgb.copy().astype(np.float32)
    color = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = (1 - alpha) * out[mask] + alpha * color
    return out.astype(np.uint8)


def draw_results(image_bgr, kept, masks, out_path: str):
    """
    kept: list of dicts [{box:(x1,y1,x2,y2), label:'food'/'plate'}]
    masks aligned with kept list indices
    """
    rgb = image_bgr[:, :, ::-1].copy()  # to RGB

    # Mask overlay first
    for item, mask in zip(kept, masks):
        if item["label"] == "food":
            rgb = overlay_mask_rgba(rgb, mask, (255, 0, 0), alpha=0.45)   # red
        else:
            rgb = overlay_mask_rgba(rgb, mask, (0, 120, 255), alpha=0.35) # blue-ish

    # Draw boxes + labels
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    for item in kept:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        color = "red" if label == "food" else "blue"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill=color)

    pil.save(out_path)
    print("Saved:", out_path)


# -----------------------------
# MAIN
# -----------------------------
def grounded_sam_food_and_plates(
    image_path: str,
    output_path: str,
    sam_checkpoint: str = "sam_vit_b_01ec64.pth",
    sam_model_type: str = "vit_b",
):
    clip_model, clip_preprocess, clip_tokenizer = load_openclip()
    dino_model = load_grounding_dino()
    sam_predictor = load_sam(sam_checkpoint, model_type=sam_model_type)

    image_bgr, boxes = dino_detect_boxes(image_path, dino_model)

    # CLIP filter boxes -> keep only food/plates
    pil_img = Image.fromarray(image_bgr[:, :, ::-1])  # RGB
    kept = []
    kept_boxes = []
    for (x1, y1, x2, y2) in boxes:
        crop = pil_img.crop((x1, y1, x2, y2))
        label = classify_clip(clip_model, clip_preprocess, clip_tokenizer, crop)
        if label is not None:
            kept.append({"box": (x1, y1, x2, y2), "label": label})
            kept_boxes.append((x1, y1, x2, y2))

    if not kept_boxes:
        print("No food/plate detections after CLIP filtering.")
        Image.fromarray(image_bgr[:, :, ::-1]).save(output_path)
        return

    masks = sam_segment(sam_predictor, image_bgr, kept_boxes)
    draw_results(image_bgr, kept, masks, output_path)


if __name__ == "__main__":
    photo_name ="rotated_90.png"
    grounded_sam_food_and_plates(
        image_path=f"photos/{photo_name}",
        output_path=f"test_output/{photo_name}_grounded_output.png",
        sam_checkpoint="models/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
    )
