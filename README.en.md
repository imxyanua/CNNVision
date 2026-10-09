# CNNVision

[中文](README.md) | [English](README.en.md) | [Tiếng Việt](README.vi.md)

A **multi-class** image classifier using a CNN (PyTorch). Class names come from dataset folders. The code is not tied to one object type.

## Status

| Piece | Status |
| ---- | ---- |
| Preprocessing, train/val/test split, CNN, `train.py` | Implemented |
| `evaluate.py` (test set + confusion matrix) | Implemented |
| `predict.py` (one image) | Implemented |
| Colab notebook | Implemented |
| CI (unittest on PRs and `main`) | Implemented |
| Loss/accuracy curves per epoch | Implemented |
| Web UI (upload an image to predict) | Implemented |
| Webcam, transfer learning, Grad-CAM | Planned |

This README does not report accuracy or benchmark numbers. Results depend on your dataset.

## Requirements

- Python 3.10+
- PyTorch, torchvision, OpenCV, Pillow, NumPy, Matplotlib, scikit-learn (`requirements.txt`)

```bash
pip install -r requirements.txt
```

If the default `torch` wheel does not match your GPU, install torch/torchvision from [pytorch.org](https://pytorch.org), then install the rest of this file.

With no discrete GPU (for example an i5 laptop), train on **Google Colab**. Local CPU still works for a small dataset.

## Dataset

Put images in `dataset/`. Do **not** commit the images.

```text
dataset/
  class_a/*.jpg
  class_b/*.jpg
```

Or a ready-made split:

```text
dataset/
  train/<class>/*
  val/<class>/*
  test/<class>/*
```

Folder name = class name. You need at least two classes, each with at least one readable image.

Default split is 70/15/15, seed 42. File lists are saved to `dataset/split.json` so later runs do not reshuffle. The test set is not used to pick the checkpoint.

## Train

```bash
python train.py --dataset dataset --epochs 10
```

Main flags: `--output` (default `models/best_model.pth`), `--output-dir` (default `outputs`), `--batch-size` (32), `--lr` (0.001), `--image-size` (224), `--color-mode` (`rgb` or `grayscale`), `--seed` (42), `--workers` (0), `--device` (`cpu` / `cuda` / omit to auto-select).

Device: CUDA when available, otherwise CPU. The best checkpoint is chosen by **validation accuracy**. The `.pth` file stores weights, `num_classes`, the class map, preprocess settings, epoch, full history, and metrics. It is gitignored. After training, `outputs/training_curves.png` shows train/val loss and accuracy.

## Evaluate

```bash
python evaluate.py --dataset dataset --checkpoint models/best_model.pth
```

Runs on the **test split** with the same preprocess as training (no augmentation). Prints accuracy, macro precision/recall/F1, and a confusion matrix.

Writes `outputs/classification_report.txt`, `outputs/metrics.json`, and `outputs/confusion_matrix.png`.

## Predict

```bash
python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth
```

Prints the class name, confidence, and per-class probabilities.

## Web UI

Requires `models/best_model.pth`. The page calls `inference.predict_image` and does not reimplement the model or preprocessing.

```bash
streamlit run app.py
```

Switch English / 中文 / Tiếng Việt. The photo fills the frame; class and confidence sit on a caption plate; the schematic is folded away. Default URL: [http://localhost:8501](http://localhost:8501). This labels the whole image, and does not draw boxes around objects.

## Google Colab

1. Open `notebooks/train_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the clone cell: `https://github.com/imxyanua/CNNVision.git`.
4. Do **not** run `pip install -r requirements.txt` on Colab. That can replace Colab’s CUDA PyTorch with a CPU wheel. Install only opencv / Pillow / matplotlib / scikit-learn as the notebook does.
5. Mount Drive or unzip the dataset, then set `DATASET`.
6. After training, download `models/best_model.pth`.

Then run `evaluate.py` / `predict.py` in the same Colab repo directory, or copy the checkpoint to your machine.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs this on CPU with tiny fake images. It does not train a real dataset.

## Layout

```text
preprocessing/   load, resize, normalize, augment (train only)
training/        split, dataloaders, train loop
models/          SimpleCNN + checkpoint
evaluation/      test-set metrics
inference/       one-image prediction
visualization/   confusion matrix, training curves, schematic network
notebooks/       Colab
train.py  evaluate.py  predict.py  app.py
```
