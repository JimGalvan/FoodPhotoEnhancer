import torch
from groundingdino.util.inference import load_image, predict

from dino.dino_utils import show_input_boxes
from dino_detector import DinoDetector
from models import load_openclip, load_grounding_dino, load_sam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUNDING_PROMPT = (
    "food, meal, cooked food, dish, plate, bowl, wooden plate, ceramic plate"
)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

if __name__ == "__main__":
    # image
    image_path = "photos/wood_dish_1.png"

    # sam
    SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE = "vit_b"

    # load models
    clip_model, preprocess, tokenizer = load_openclip()
    dino_model = load_grounding_dino()
    sam_predictor = load_sam(SAM_CHECKPOINT, SAM_MODEL_TYPE)

    # create objects
    dino_detector = DinoDetector(
        device=DEVICE,
        model=dino_model
    )

    # detect objects
    image_source, input_boxes = dino_detector.detect_boxes(
        image_path,
        grounding_prompt=GROUNDING_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    show_input_boxes(image_source, input_boxes)





