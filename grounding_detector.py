import torch
import open_clip
import cv2
import numpy as np
from PIL import Image, ImageDraw
from groundingdino.util.inference import load_model, load_image, predict


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT = "food"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
CLIP_THRESHOLD = 0.30


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


# -----------------------------
# CLIP FOOD CHECK
# -----------------------------
def clip_is_food(clip_model, preprocess, tokenizer, crop_img):
    image = preprocess(crop_img).unsqueeze(0).to(DEVICE)
    text = tokenizer(["food", "not food"]).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (100 * image_features @ text_features.T).softmax(dim=-1)

    return probs[0, 0].item() > CLIP_THRESHOLD


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def detect_food(image_path, output_path="food_detected.jpg"):
    # Load models
    clip_model, clip_preprocess, clip_tokenizer = load_openclip()
    dino_model = load_grounding_dino()

    # Load image
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    # Grounded detection
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    pil_img = Image.fromarray(image_source)
    draw = ImageDraw.Draw(pil_img)

    # Process each box
    for box in boxes:
        x1 = int((box[0] - box[2] / 2) * w)
        y1 = int((box[1] - box[3] / 2) * h)
        x2 = int((box[0] + box[2] / 2) * w)
        y2 = int((box[1] + box[3] / 2) * h)

        crop = pil_img.crop((x1, y1, x2, y2))

        # CLIP verification
        if clip_is_food(clip_model, clip_preprocess, clip_tokenizer, crop):
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, y1 - 10), "food", fill="red")

    pil_img.save(output_path)
    print(f"Saved result to {output_path}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    detect_food("photos/wood_dish_1.png", "test_output/output_food.jpg")
