import torch
import clip


class PreTrainedModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        model, preprocess = clip.load(self.model_name, jit=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, preprocess
