import torch
from pathlib import Path

from PIL import Image
from torchvision import transforms

from model import CatDogCNN


# ======================
# 路径配置
# ======================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 类别名称
CLASS_NAMES = ["cat", "dog"]


def load_model():
    """
    加载训练好的模型
    """

    model = CatDogCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


def predict_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    model = load_model()

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    class_name = CLASS_NAMES[
        predicted.item()
    ]

    confidence = confidence.item() * 100

    return class_name, confidence


def main():

    try:

        image_path = input(
            "请输入图片路径："
        )

        class_name, confidence = predict_image(
            image_path
        )

        print("\n===== 预测结果 =====")
        print(f"类别: {class_name}")
        print(f"置信度: {confidence:.2f}%")

    except Exception as e:

        print("预测失败")
        print(e)


if __name__ == "__main__":
    main()