import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model import create_model
# ======================
# 路径配置
# ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
CLASS_NAMES = [
    "cat",
    "dog"
]
# ======================
# 加载模型
# ======================
def load_model():
    model = create_model()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )
    model.to(DEVICE)
    model.eval()
    return model
# ======================
# 图片预处理
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ======================
# 单张图片预测
# ======================
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
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        confidence, predicted = torch.max(
            probabilities,
            1
        )
    label = CLASS_NAMES[
        predicted.item()
    ]
    confidence = (
        confidence.item() * 100
    )
    return label, confidence
# ======================
# 主函数
# ======================
def main():
    try:
        image_path = input(
            "请输入图片路径："
        )
        label, confidence = predict_image(
            image_path
        )
        print("\n===== 预测结果 =====")
        print(
            f"预测类别: {label}"
        )
        print(
            f"置信度: "
            f"{confidence:.2f}%"
        )
    except Exception as e:
        print("预测失败")
        print(e)

if __name__ == "__main__":
    main()