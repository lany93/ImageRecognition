"""Read data."""

from pathlib import Path
import json
import torch
import clip

IMAGE_LABEL = "image_label"
IMAGE_CAPTION = "caption"
IMAGE_PATH = "image_path"
IMAGE_LABEL_TOKEN = "label_token"
DATASET_SPLIT = ["train", "val", "test"]
LABELS = "labels"


class ImageLoader:
    def __init__(self, image_base_path: Path, dataset_type: str):
        """
        Args:
            image_base_path (Path):     Path to the image folder.
            json_base_path (Path):      Path to the json files.
        """
        self.base_path = image_base_path
        self.dataset_type = dataset_type
        self.dataset = {}

    def create_indo_fashion_dataset(self, dataset_name: str = "indo_fashion"):
        """
        Load indo fashion images.
        """
        indo_fashion_folder_path = self.base_path / dataset_name
        json_path = (
            self.base_path / f"{dataset_name}/images/{self.dataset_type}_data.json"
        )

        input_data = []
        with open(json_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                input_data.append(obj)

        self.dataset[IMAGE_PATH] = [
            indo_fashion_folder_path / info["image_path"] for info in input_data
        ]

        self.dataset[IMAGE_CAPTION] = [item["product_title"] for item in input_data]
        self.dataset[IMAGE_LABEL] = [item["class_label"] for item in input_data]
        self.dataset[IMAGE_LABEL_TOKEN] = self._tokenize_labels()
        self.dataset[LABELS] = list(set(self.dataset[IMAGE_LABEL]))

    def _tokenize_labels(self) -> dict:
        """Tokenize labels. Note that this requires all the labels exists at least ONCE in all the dataset, e.g. train, val and test."""
        if IMAGE_LABEL in self.dataset:
            labels = list(set(self.dataset[IMAGE_LABEL]))
            return dict(
                zip(
                    labels,
                    torch.cat([clip.tokenize(f"a photo of {lab}") for lab in labels]),
                )
            )
        else:
            raise ValueError("No labels in dataset. Create dataset first.")

    def create_food_101_dataset(self):
        """
        Load food 101 images.
        """
        pass
