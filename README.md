# Cat Dog Classification Based on Deep Learning

基于深度学习的猫狗图像分类系统。

本项目使用 PyTorch 框架实现，实现了两种图像分类模型：

-   自定义 CNN 网络（Custom CNN）
-   ResNet50 迁移学习模型（Transfer Learning）

实现从输入图片到 Cat / Dog 分类结果的完整流程。

------------------------------------------------------------------------

# 1. 项目简介

本项目针对猫狗图像二分类任务，完成数据预处理、模型训练、模型保存以及图片预测。

主要功能：

-   图像数据加载与增强
-   CNN 特征提取
-   模型训练
-   模型测试
-   图片分类预测

------------------------------------------------------------------------

# 2. 算法流程

整体流程：

    输入图片
        |
        ↓
    数据预处理
        |
        ↓
    数据增强
    (RandomFlip / Rotation / ColorJitter)
        |
        ↓
    CNN特征提取
        |
        ↓
    Feature Map
        |
        ↓
    全连接分类层
        |
        ↓
    Softmax分类
        |
        ↓
    输出 Cat / Dog

------------------------------------------------------------------------

# 3. 模型介绍

## 3.1 自定义 CNN

网络结构：

    Input Image

    ↓

    Conv2d

    ↓

    BatchNorm + ReLU

    ↓

    MaxPooling

    ↓

    多层卷积特征提取

    ↓

    Adaptive Average Pooling

    ↓

    Fully Connected Layer

    ↓

    Classification

特点：

-   参数量较少
-   训练速度较快
-   适合作为基础模型

## 3.2 ResNet50

采用 ImageNet 预训练 ResNet50。

流程：

    Input

    ↓

    Pretrained ResNet50

    ↓

    Feature Extraction

    ↓

    Replace FC Layer

    ↓

    Binary Classification

利用迁移学习提高模型特征提取能力。

------------------------------------------------------------------------

# 4. 项目结构

    cat_dog_classification

    ├── README.md
    ├── requirements.txt

    └── src

        ├── dataset.py

        ├── defined_model.py
        ├── defined_train.py
        ├── defined_predict.py

        ├── model.py
        ├── train.py
        ├── predict.py

        └── utils.py

文件说明：

  文件                 功能
  -------------------- ---------------
  defined_model.py     自定义CNN模型
  defined_train.py     CNN训练
  defined_predict.py   CNN预测
  model.py             ResNet50模型
  train.py             ResNet50训练
  predict.py           图片预测
  dataset.py           数据处理

------------------------------------------------------------------------

# 5. 环境配置

Python:

    Python >= 3.10

主要依赖：

    PyTorch
    Torchvision
    OpenCV
    NumPy

安装：

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 6. 参数设置

训练参数：

  参数            设置
  --------------- ------------------
  Image Size      224×224
  Batch Size      32
  Optimizer       Adam
  Learning Rate   0.001
  Loss Function   CrossEntropyLoss
  Epoch           50

数据增强：

-   RandomHorizontalFlip
-   RandomRotation
-   ColorJitter

------------------------------------------------------------------------

# 7. 数据集

数据类别：

    Cat
    Dog

目录：

    data/

    ├── train/

    │   ├── cat

    │   └── dog


    └── val/

        ├── cat

        └── dog

------------------------------------------------------------------------

# 8. 训练方法

## 自定义 CNN

运行：

``` bash
python src/defined_train.py
```

## ResNet50

运行：

``` bash
python src/train.py
```

训练完成后模型保存到 models 文件夹。

------------------------------------------------------------------------

# 9. 图片预测

运行：

``` bash
python src/defined_predict.py
```

或：

``` bash
python src/predict.py
```

输出示例：

    Prediction: Cat

    Confidence: 0.98

------------------------------------------------------------------------

# 10. 实验结果

实验比较：

  模型         方法
  ------------ ----------
  Custom CNN   从零训练
  ResNet50     迁移学习

结果表明：

-   自定义 CNN 可以完成猫狗分类任务
-   ResNet50 利用预训练特征，具有更好的泛化能力

------------------------------------------------------------------------

# 11. 复现步骤

## Step 1 下载项目

``` bash
git clone https://github.com/velocity-guest/cat_dog_classification.git
```

## Step 2 安装环境

``` bash
pip install -r requirements.txt
```

## Step 3 准备数据集

将数据放入：

    data/

## Step 4 训练模型

CNN:

``` bash
python src/defined_train.py
```

ResNet50:

``` bash
python src/train.py
```

## Step 5 预测

``` bash
python src/predict.py
```

------------------------------------------------------------------------

# Author

# Author

Author: 赵超艺

Project: Cat Dog Classification Based on Deep Learning