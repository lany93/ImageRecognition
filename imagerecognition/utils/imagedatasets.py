"""Image pairs dataset."""

from torch.utils.data import Dataset
from PIL import Image


class ImageTitleDataset(Dataset):
    def __init__(self, list_image_path, list_txt, processor):
        """
        Args:
            list_image_path (list): List of image file paths.
            list_txt (list): List of corresponding text titles.
            processor (CLIPProcessor): Processor for preprocessing images and text.
        """
        self.image_path = list_image_path
        self.title = list_txt
        self.processor = processor  # Hugging Face CLIP processor

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.title)

    def __getitem__(self, idx):
        """Returns a preprocessed image and its corresponding tokenized text."""
        image = Image.open(self.image_path[idx]).convert(
            "RGB"
        )  # Ensure image is in RGB
        text = self.title[idx]

        # Preprocess image and tokenize text using the processor
        inputs = self.processor(
            text=[text], images=image, return_tensors="pt", padding=True
        )

        # Return preprocessed image and text
        return inputs["pixel_values"].squeeze(0), inputs["input_ids"].squeeze(0)
