import torch
from PIL import Image


class ClipClassifier:
    def __init__(self, model, preprocess, tokenizer, device):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device

    def classify_dish_items(
            self,
            crop: Image.Image,
            food_prompts: list,
            container_prompts: list,
            threshold=0.30
    ):
        dish_prompts = food_prompts + container_prompts + ["not food"]
        image = self.preprocess(crop).unsqueeze(0).to(self.device)
        text = self.tokenizer(dish_prompts).to(self.device)

        with torch.no_grad():
            img_f = self.model.encode_image(image)
            txt_f = self.model.encode_text(text)

            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)

            probs = (100 * img_f @ txt_f.T).softmax(dim=-1)

        idx = int(probs.argmax().item())
        score = float(probs[0, idx].item())
        label = dish_prompts[idx]

        if score < threshold:
            return None
        if label in food_prompts:
            return "food"
        if label in container_prompts:
            return "plate"
        return None
