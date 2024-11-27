"""Read data."""

from pathlib import Path
import json


class ImageLoader:
    def __init__(self, image_base_path: Path, dataset_type: str):
        """
        Args:
            image_base_path (Path):     Path to the image folder.
            json_base_path (Path):      Path to the json files.
        """
        self.base_path = image_base_path
        self.dataset_type = dataset_type
        self.list_image_path = None
        self.list_text = None

    def load_indo_fashion(self):
        """
        Load indo fashion images.
        """
        self.list_image_path = []
        self.list_text = []
        self.input_data = []
        json_path = self.base_path / f"indo_fashion/{self.dataset_type}_data.json"

        if self.dataset_type == "test":
            data_path = json_path / "test"
        elif self.dataset_type == "train":
            data_path = json_path / "test"
        else:
            data_path = json_path / "val"

        with open(json_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.input_data.append(obj)

        for item in self.input_data:
            img_path = data_path / item["image_path"].split("/")[-1]

            # As we have image text pair, we use product title as description.
            caption = item["product_title"][:40]
            self.list_image_path.append(img_path)
            self.list_text.append(caption)

        def load_food_101(self):
            """
            Load food 101 images.
            """
            pass
