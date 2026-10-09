# CNNVision

[中文](README.md) | [English](README.en.md) | [Tiếng Việt](README.vi.md)

使用 CNN（PyTorch）的**多类别**图像分类系统。类别名来自数据集文件夹，不绑定某一种物体。

## 状态

| 模块 | 状态 |
| ---- | ---- |
| 预处理、train/val/test 划分、CNN、`train.py` | 已实现 |
| `evaluate.py`（测试集 + 混淆矩阵） | 已实现 |
| `predict.py`（单张图片） | 已实现 |
| Colab 笔记本 | 已实现 |
| CI（PR 与 `main` 上的 unittest） | 已实现 |
| 按 epoch 的 loss/accuracy 曲线 | 已实现 |
| Web UI（上传图片预测） | 已实现 |
| 摄像头、迁移学习、Grad-CAM | 计划中 |

README 不提供准确率或基准数字。结果取决于你的数据集。

## 环境

- Python 3.10+
- PyTorch、torchvision、OpenCV、Pillow、NumPy、Matplotlib、scikit-learn（见 `requirements.txt`）

```bash
pip install -r requirements.txt
```

若默认 `torch` 轮子与 GPU 不匹配，先从 [pytorch.org](https://pytorch.org) 安装 torch/torchvision，再安装本文件其余依赖。

没有独立 GPU（例如 i5 笔记本）时，请在 **Google Colab** 上训练。本机 CPU 仍可跑小数据集。

## 数据集

把图片放到 `dataset/`。图片**不要**提交到 git。

```text
dataset/
  class_a/*.jpg
  class_b/*.jpg
```

或已划分好的目录：

```text
dataset/
  train/<class>/*
  val/<class>/*
  test/<class>/*
```

文件夹名 = 类别名。至少两个类别，每个类别至少有一张可读图片。

默认划分 70/15/15，seed 42，文件列表写入 `dataset/split.json`，避免下次训练重新打乱。测试集不用于选择 checkpoint。

## 训练

```bash
python train.py --dataset dataset --epochs 10
```

主要参数：`--output`（默认 `models/best_model.pth`）、`--output-dir`（默认 `outputs`）、`--batch-size`（32）、`--lr`（0.001）、`--image-size`（224）、`--color-mode`（`rgb` 或 `grayscale`）、`--seed`（42）、`--workers`（0）、`--device`（`cpu` / `cuda` / 省略则自动选择）。

设备：有 CUDA 用 CUDA，否则 CPU。按 **验证集 accuracy** 保存最佳 checkpoint。`.pth` 含 weights、`num_classes`、类别映射、预处理配置、epoch、完整 history、指标。该文件已被 gitignore。训练结束后写入 `outputs/training_curves.png`（train/val 的 loss 与 accuracy）。

## 评估

```bash
python evaluate.py --dataset dataset --checkpoint models/best_model.pth
```

在 **test split** 上评估，预处理与训练一致（无增强）。输出 accuracy、macro precision/recall/F1、混淆矩阵。

写入 `outputs/classification_report.txt`、`outputs/metrics.json`、`outputs/confusion_matrix.png`。

## 预测

```bash
python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth
```

打印类别名、置信度，以及每个类别的概率。

## Web UI

需要已有 `models/best_model.pth`。界面调用 `inference.predict_image`，不重新实现模型或预处理。

```bash
streamlit run app.py
```

页面可切换 中文 / English / Tiếng Việt。左侧设置与图片，右侧示意图始终显示。默认打开 [http://localhost:8501](http://localhost:8501)。这是整图单标签分类，不会框出图中每个物体。

## Google Colab

1. 打开 `notebooks/train_colab.ipynb`。
2. Runtime → Change runtime type → **T4 GPU**。
3. 运行 clone 单元格：`https://github.com/imxyanua/CNNVision.git`。
4. **不要**在 Colab 上执行 `pip install -r requirements.txt`，否则会把 Colab 的 CUDA 版 PyTorch 换成 CPU 轮子。只需按笔记本安装 opencv / Pillow / matplotlib / scikit-learn。
5. 挂载 Drive 或解压数据集，修改 `DATASET`。
6. 训练完成后下载 `models/best_model.pth`。

训练之后可在 Colab 同一仓库目录运行 `evaluate.py` / `predict.py`，或把 checkpoint 拷回本机再运行。

## 测试

```bash
python -m unittest discover -s tests -v
```

GitHub Actions 在 CPU 上用合成小图运行该命令，不会训练真实数据集。

## 目录

```text
preprocessing/   读取、缩放、归一化、增强（仅训练集）
training/        划分、dataloader、训练循环
models/          SimpleCNN + checkpoint
evaluation/      测试集指标
inference/       单张图片预测
visualization/   混淆矩阵、训练曲线、示意神经网络
notebooks/       Colab
train.py  evaluate.py  predict.py  app.py
```
