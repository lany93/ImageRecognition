"""Main file."""

from pathlib import Path
from utils.pipeline import Pipeline
from utils.read_data import ImageLoader
import ssl
from utils.model import PreTrainedModel, FineTuneModel
from torch.utils.data import DataLoader
from utils.read_data import IMAGE_LABEL
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

IMAGE_BASE_PATH = Path("images")
JSON_BASE_PATH = Path()
MODEL_NAME = "ViT-B/32"
BATCH_SIZE = 64
SHUFFLE_TRAIN = True
SHUFFLE_TEST = False
LEARNING_RATE = 1e-4


def main():
    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # Allows to download the pre-trained model from huggingface
    # Load Images
    image_train = ImageLoader(image_base_path=IMAGE_BASE_PATH, dataset_type="train")
    image_train.create_indo_fashion_dataset()

    image_val = ImageLoader(image_base_path=IMAGE_BASE_PATH, dataset_type="val")
    image_val.create_indo_fashion_dataset()

    # Load Model
    pretrained_model = PreTrainedModel(MODEL_NAME)
    model, preprocess, device = pretrained_model.load_model()
    # tokenizer = pretrained_model.load_tokenizer()

    # Data Loader
    train_loader = DataLoader(
        dataset=Pipeline(image_train.dataset),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
    )
    val_loader = DataLoader(
        dataset=Pipeline(image_val.dataset), batch_size=BATCH_SIZE, shuffle=SHUFFLE_TEST
    )

    # Instantiate Fine Tune Model
    num_classes = len(set(image_train.dataset[IMAGE_LABEL]))
    model_fine_tuned = FineTuneModel(model=model, num_classes=num_classes).to(
        device=device
    )

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_fine_tuned.classifier.parameters(), lr=LEARNING_RATE)

    # Fine Tune
    num_epochs = 5

    for epoch in range(num_epochs):
        model_fine_tuned.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_fine_tuned(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}"
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )

        model_fine_tuned.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_fine_tuned(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total}%")
    torch.save(model_fine_tuned.state_dict(), "clip_finetuned.pth")


if __name__ == "__main__":
    main()
