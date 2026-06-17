import torch
from pathlib import Path

from model import create_model


# ======================
# 路径配置
# ======================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "best_model.pth"
)

QUANTIZED_MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "best_model_quantized.pth"
)


# ======================
# 设备
# ======================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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

    model.to("cpu")

    model.eval()

    return model



# ======================
# 模型量化
# ======================

def quantize_model(model):

    print("开始模型量化...")


    quantized_model = torch.quantization.quantize_dynamic(
        model,

        {
            torch.nn.Linear
        },

        dtype=torch.qint8
    )


    return quantized_model



# ======================
# 主函数
# ======================

def main():

    print(
        "Loading model..."
    )


    model = load_model()


    print(
        "Original model loaded"
    )


    quantized_model = quantize_model(
        model
    )


    torch.save(
        quantized_model.state_dict(),
        QUANTIZED_MODEL_PATH
    )


    print(
        "Quantization finished!"
    )


    print(
        "Saved:"
    )

    print(
        QUANTIZED_MODEL_PATH
    )



if __name__ == "__main__":

    main()