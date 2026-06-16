import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    """
    猫狗分类卷积神经网络
    输入尺寸: 3×224×224
    输出尺寸: 2
    """

    def __init__(self):
        super(CatDogCNN, self).__init__()

        # 特征提取部分
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256 * 14 * 14,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    try:

        model = CatDogCNN()

        x = torch.randn(
            4,
            3,
            224,
            224
        )

        y = model(x)

        print("模型结构测试成功")
        print("输入尺寸:", x.shape)
        print("输出尺寸:", y.shape)

    except Exception as e:

        print("模型测试失败")
        print(e)