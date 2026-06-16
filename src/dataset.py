import shutil
import random
import logging
from pathlib import Path

# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======================
# 项目根目录
# ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据目录
RAW_DIR = PROJECT_ROOT / "data" / "train"

# 处理后数据目录
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 训练集比例
TRAIN_RATIO = 0.8


def create_dirs():
    """
    创建目录结构
    """
    dirs = [
        PROCESSED_DIR / "train" / "cat",
        PROCESSED_DIR / "train" / "dog",
        PROCESSED_DIR / "val" / "cat",
        PROCESSED_DIR / "val" / "dog"
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    logging.info("目录创建完成")


def split_files():
    """
    划分训练集和验证集
    """

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"数据目录不存在:\n{RAW_DIR}"
        )

    cat_files = []
    dog_files = []

    for file in RAW_DIR.iterdir():

        if not file.is_file():
            continue

        if file.name.startswith("cat"):
            cat_files.append(file)

        elif file.name.startswith("dog"):
            dog_files.append(file)

    random.shuffle(cat_files)
    random.shuffle(dog_files)

    cat_split = int(len(cat_files) * TRAIN_RATIO)
    dog_split = int(len(dog_files) * TRAIN_RATIO)

    train_cat = cat_files[:cat_split]
    val_cat = cat_files[cat_split:]

    train_dog = dog_files[:dog_split]
    val_dog = dog_files[dog_split:]

    logging.info(f"猫图片总数: {len(cat_files)}")
    logging.info(f"狗图片总数: {len(dog_files)}")

    return train_cat, val_cat, train_dog, val_dog


def copy_files(files, target_dir):
    """
    复制图片
    """

    for file in files:

        try:
            shutil.copy(file, target_dir / file.name)

        except Exception as e:
            logging.error(f"复制失败: {file.name}")
            logging.error(str(e))


def main():

    try:

        create_dirs()

        train_cat, val_cat, train_dog, val_dog = split_files()

        copy_files(
            train_cat,
            PROCESSED_DIR / "train" / "cat"
        )

        copy_files(
            val_cat,
            PROCESSED_DIR / "val" / "cat"
        )

        copy_files(
            train_dog,
            PROCESSED_DIR / "train" / "dog"
        )

        copy_files(
            val_dog,
            PROCESSED_DIR / "val" / "dog"
        )

        print("\n===== 数据集划分完成 =====")

        print(f"Train Cat : {len(train_cat)}")
        print(f"Val Cat   : {len(val_cat)}")

        print(f"Train Dog : {len(train_dog)}")
        print(f"Val Dog   : {len(val_dog)}")

        print(f"\n输出目录:")
        print(PROCESSED_DIR)

        logging.info("数据集划分完成")

    except Exception as e:

        logging.error("程序执行失败")
        logging.error(str(e))

        print("\n发生错误:")
        print(e)


if __name__ == "__main__":
    main()