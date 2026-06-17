import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


def create_model():

    model = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.5),

        nn.Linear(
            model.fc.in_features,
            2
        )
    )

    return model


if __name__ == "__main__":

    model = create_model()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("ResNet50加载成功")
    print(f"参数量: {total_params:,}")