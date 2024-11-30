import torch
import clip
import torch.nn as nn


class PreTrainedModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        model, preprocess = clip.load(self.model_name, jit=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, preprocess, device

    # def load_tokenizer(self):
    #     return CLIPTokenizer.from_pretrained(self.model_name)


# Modify the model to include a classifier for subcategories
class FineTuneModel(nn.Module):
    def __init__(self, model, num_classes):
        super(FineTuneModel, self).__init__()
        self.model = model
        # self.image_text_combiner = nn.Linear(
        #     model.visual.output_dim + model.textual.output_dim, model.visual.output_dim
        # )
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
            # text_features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)
