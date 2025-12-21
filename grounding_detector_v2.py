import torch
import open_clip
from PIL import Image, ImageDraw
from groundingdino.util.inference import load_model, load_image, predict

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_PROMPT = (
    "food, meal, dish, plate, bowl, wooden plate, ceramic plate, food bowl"
)

FOOD_TEXTS = [
    "food",
    "meal",
    "dish",
    "cooked food",
]

CONTAINER_TEXTS = [
    "plate",
    "bowl",
    "dish",
    "wooden plate",
    "ceramic plate",
    "food bowl",
]

CLIP_THRESHOLD = 0.30


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
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



# --------------------------------------------------
# CLIP CLASSIFICATION
# --------------------------------------------------
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
        return None, None

    if label in FOOD_TEXTS:
        return "food", "red"
    elif label in CONTAINER_TEXTS:
        return "plate", "blue"
    else:
        return None, None


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def detect_food_and_plates(image_path, output_path):
    clip_model, preprocess, tokenizer = load_openclip()
    dino_model = load_grounding_dino()

    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    boxes, _, _ = predict(
        model=dino_model,
        image=image,
        caption=GROUNDING_PROMPT,
        box_threshold=0.25,
        text_threshold=0.25,
        device=DEVICE,
    )

    pil_img = Image.fromarray(image_source)
    draw = ImageDraw.Draw(pil_img)

    for box in boxes:
        cx, cy, bw, bh = box.tolist()

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        crop = pil_img.crop((x1, y1, x2, y2))

        label, color = classify_clip(
            clip_model, preprocess, tokenizer, crop
        )

        if label is None:
            continue

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, y1 - 12), label, fill=color)

    pil_img.save(output_path)
    print("Saved result to:", output_path)


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    photo_name = "ok_light_1.png"
    detect_food_and_plates(
        image_path=f"photos/{photo_name}",
        output_path=f"test_output/{photo_name}_detected.jpg",
    )
