import torch

from pathlib import Path

from PIL import Image

from torchvision import transforms

from defined_model import DefinedCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "defined_cnn_best.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "cat",
    "dog"
]


transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


def load_model():

    model = DefinedCNN()

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

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    model = load_model()

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    return (
        CLASS_NAMES[pred.item()],
        confidence.item() * 100
    )


if __name__ == "__main__":

    image_path = input(
        "请输入图片路径："
    )

    label, confidence = predict_image(
        image_path
    )

    print("\n===== 预测结果 =====")

    print("类别:", label)

    print(
        f"置信度: "
        f"{confidence:.2f}%"
    )