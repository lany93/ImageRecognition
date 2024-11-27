"""Main file."""

from pathlib import Path
from utils.read_data import ImageLoader
import ssl
from utils.model import PreTrainedModel

IMAGE_BASE_PATH = Path("images")
JSON_BASE_PATH = Path()
MODEL_NAME = "ViT-B/32"


def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    # Load Images
    image_train = ImageLoader(image_base_path=IMAGE_BASE_PATH, dataset_type="train")
    image_train.load_indo_fashion()

    image_test = ImageLoader(image_base_path=IMAGE_BASE_PATH, dataset_type="test")
    image_test.load_indo_fashion()

    # Load Model
    pretrained_model = PreTrainedModel(MODEL_NAME)
    model, preprocess = pretrained_model.load_model()

    # Train-Test Split
    # train_loader = DataLoader(
    #     Pipeline(image_train.dataset), batch_size=32, shuffle=True
    # )
    # test_loader = DataLoader(Pipeline(image_test.dataset), batch_size=32, shuffle=False)


if __name__ == "__main__":
    main()
