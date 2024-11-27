"""Read data."""

from pathlib import Path
import json
import os


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

    def load_indo_fashion(self, dataset_name: str = "indo_fashion"):
        """
        Load indo fashion images.
        """
        self.dataset = {}
        indo_fashion_folder_path = self.base_path / dataset_name
        json_path = self.base_path / f"{dataset_name}/{self.dataset_type}_data.json"

        if self.dataset_type == "test":
            image_path = indo_fashion_folder_path / "test"
        elif self.dataset_type == "train":
            image_path = indo_fashion_folder_path / "test"
        else:
            image_path = indo_fashion_folder_path / "val"

        input_data = []
        with open(json_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                input_data.append(obj)

        self.dataset["image_path"] = [
            image_path / path for path in os.listdir(image_path)
        ]
        self.dataset["caption"] = [item["product_title"] for item in input_data]
        self.dataset["label"] = [item["class_label"] for item in input_data]

    def load_food_101(self):
        """
        Load food 101 images.
        """
        pass
