"""Main file."""

from pathlib import Path
from utils.read_data import ImageLoader

IMAGE_BASE_PATH = Path("images")
JSON_BASE_PATH = Path()


def main():
    images = ImageLoader(image_base_path=IMAGE_BASE_PATH, dataset_type="train")
    images.load_indo_fashion()
    print(len(images.list_image_path))


if __name__ == "__main__":
    main()
