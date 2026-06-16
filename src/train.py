import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CatDogCNN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "val"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pth"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


def evaluate(model, val_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Classes:", train_dataset.classes)

    model = CatDogCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_acc = 0.0

    try:

        for epoch in range(EPOCHS):

            model.train()

            running_loss = 0.0

            print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

            for batch_idx, (images, labels) in enumerate(train_loader):

                if batch_idx % 20 == 0:
                    print(
                        f"Epoch {epoch + 1} | "
                        f"Batch {batch_idx}/{len(train_loader)}"
                    )

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            val_acc = evaluate(
                model,
                val_loader,
                device
            )

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Loss: {avg_loss:.4f} "
                f"Val Acc: {val_acc:.2f}%"
            )

            logging.info(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Loss={avg_loss:.4f} "
                f"ValAcc={val_acc:.2f}"
            )

            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    model.state_dict(),
                    MODEL_PATH
                )

                print(
                    f"Best model saved "
                    f"({best_acc:.2f}%)"
                )

    except Exception as e:

        logging.error(str(e))

        print("Training Error:")
        print(e)

    print("\nTraining Finished")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Model Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()