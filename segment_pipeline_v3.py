if __name__ == "__main__":
    # load models
    clip_model, preprocess, tokenizer = load_openclip()
    dino_model = load_grounding_dino()
    sam_predictor = load_sam(sam_checkpoint, sam_model_type)