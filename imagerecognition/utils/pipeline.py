"""Image pairs dataset."""

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class Pipeline(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        label = self.data["label"][idx]
        with Image.open(self.data["image_path"][idx]) as img:
            return self.transform(img), label
