import os

import open_clip
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor


def load_openclip(device: str = "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def load_grounding_dino():
    return load_model(
        "dino/config/GroundingDINO_SwinT_OGC.py",
        "checkpoints/groundingdino_swint_ogc.pth",
    )


def load_sam(checkpoint_path: str, model_type: str = "vit_b", device: str = "cpu") -> SamPredictor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)


def load_depth_anything_v2(device):
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384]
    )

    model.load_state_dict(
        torch.load(
            "checkpoints/depth_anything_v2_vits.pth",
            map_location="cpu"
        )
    )

    return model.to(device).eval()