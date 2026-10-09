# CNNVision

[中文](README.md) | [English](README.en.md) | [Tiếng Việt](README.vi.md)

Hệ thống phân loại ảnh **nhiều lớp** bằng CNN (PyTorch). Tên class lấy từ thư mục dataset, không gắn chết một loại đối tượng.

## Trạng thái

| Phần | Trạng thái |
| ---- | ---------- |
| Preprocessing, split train/val/test, CNN, `train.py` | Implemented |
| `evaluate.py` (test set + confusion matrix) | Implemented |
| `predict.py` (một ảnh) | Implemented |
| Notebook Colab | Implemented |
| CI (unittest trên PR và `main`) | Implemented |
| Biểu đồ loss/accuracy theo epoch | Implemented |
| Web UI (upload ảnh để đoán) | Implemented |
| Webcam, transfer learning, Grad-CAM | Planned |

Không có số accuracy hay benchmark trong README. Kết quả phụ thuộc dataset của bạn.

## Yêu cầu

- Python 3.10+
- PyTorch, torchvision, OpenCV, Pillow, NumPy, Matplotlib, scikit-learn (`requirements.txt`)

```bash
pip install -r requirements.txt
```

Nếu wheel `torch` mặc định không khớp GPU, cài torch/torchvision từ [pytorch.org](https://pytorch.org) rồi cài nốt file này.

Không GPU (ví dụ i5 laptop): train trên **Google Colab**. Máy local vẫn chạy CPU được với bộ nhỏ.

## Dataset

Đặt ảnh vào `dataset/`. Ảnh **không** commit lên git.

```text
dataset/
  class_a/*.jpg
  class_b/*.jpg
```

Hoặc đã chia sẵn:

```text
dataset/
  train/<class>/*
  val/<class>/*
  test/<class>/*
```

Tên folder = tên class. Cần ít nhất hai class, mỗi class có ảnh đọc được.

Split mặc định 70/15/15, seed 42, danh sách file lưu `dataset/split.json` để lần sau không xáo lại. Test không dùng để chọn checkpoint.

## Train

```bash
python train.py --dataset dataset --epochs 10
```

Cờ chính: `--output` (mặc định `models/best_model.pth`), `--output-dir` (mặc định `outputs`), `--batch-size` (32), `--lr` (0.001), `--image-size` (224), `--color-mode` (`rgb` hoặc `grayscale`), `--seed` (42), `--workers` (0), `--device` (`cpu` / `cuda` / bỏ trống = tự chọn).

Device: CUDA nếu có, không thì CPU. Checkpoint tốt nhất theo **validation accuracy**. File `.pth` gồm weights, `num_classes`, class map, preprocess, epoch, toàn bộ history, metric. File này gitignored. Sau train ghi `outputs/training_curves.png` (loss/accuracy train và val).

## Evaluate

```bash
python evaluate.py --dataset dataset --checkpoint models/best_model.pth
```

Chạy trên **test split**, cùng preprocess lúc train (không augment). In accuracy, precision/recall/F1 macro, confusion matrix.

Ghi `outputs/classification_report.txt`, `outputs/metrics.json`, `outputs/confusion_matrix.png`.

## Predict

```bash
python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth
```

In tên class, confidence, và xác suất từng class.

## Web UI

Cần có `models/best_model.pth`. Trang gọi `inference.predict_image`, không viết lại model hay preprocess.

```bash
streamlit run app.py
```

Upload ảnh: trái là ảnh, phải là class và độ tin cậy, dưới là sơ đồ nơ-ron minh họa (nốt lớp ra sáng theo Softmax, không phải activation thật trong CNN). Mặc định: [http://localhost:8501](http://localhost:8501). Đây là gắn một nhãn cho cả tấm, không khoanh từng vật.

## Google Colab

1. Mở `notebooks/train_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Chạy cell clone `https://github.com/imxyanua/CNNVision.git`.
4. **Không** `pip install -r requirements.txt` trên Colab — sẽ thay bản CUDA của Colab bằng wheel CPU. Chỉ cài opencv / Pillow / matplotlib / scikit-learn như notebook.
5. Gắn Drive hoặc unzip dataset, sửa biến `DATASET`.
6. Train, rồi tải `models/best_model.pth`.

Evaluate / predict sau khi train: chạy `evaluate.py` / `predict.py` trong cùng thư mục repo trên Colab, hoặc về máy với checkpoint đã tải.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

CI GitHub Actions chạy lệnh này trên CPU với ảnh giả, không train dataset thật.

## Cấu trúc

```text
preprocessing/   load, resize, normalize, augment (chỉ train)
training/        split, dataloader, vòng train
models/          SimpleCNN + checkpoint
evaluation/      metric trên test
inference/       đoán một ảnh
visualization/   confusion matrix, training curves, sơ đồ nơ-ron
notebooks/       Colab
train.py  evaluate.py  predict.py  app.py
```
