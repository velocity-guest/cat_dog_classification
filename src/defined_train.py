import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

from defined_model import DefinedCNN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"

VAL_DIR = PROJECT_ROOT / "data" / "processed" / "val"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "defined_cnn_best.pth"

BATCH_SIZE = 64

EPOCHS = 10

LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", DEVICE)


train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
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


model = DefinedCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total


best_acc = 0.0

try:

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        print(
            f"\n===== Epoch "
            f"{epoch + 1}/{EPOCHS} ====="
        )

        for batch_idx, (
            images,
            labels
        ) in enumerate(train_loader):

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:

                print(
                    f"Batch "
                    f"{batch_idx}/"
                    f"{len(train_loader)}"
                )

        scheduler.step()

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        val_acc = evaluate()

        print(
            f"Loss={avg_loss:.4f} "
            f"ValAcc={val_acc:.2f}%"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"Best Model Saved "
                f"({best_acc:.2f}%)"
            )

except Exception as e:

    logging.error(str(e))

    print(e)

print("\nTraining Finished")

print(f"Best Accuracy: {best_acc:.2f}%")

print(f"Model Saved: {MODEL_PATH}")