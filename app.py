from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from inference.predict import predict_image

st.set_page_config(page_title="CNNVision", layout="centered")
st.title("CNNVision")
st.caption("多类别图像分类 · 使用已训练的 checkpoint，不重新训练。")

checkpoint = st.text_input("Checkpoint", value="models/best_model.pth")
uploaded = st.file_uploader(
    "上传图片",
    type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"],
)

if uploaded is None:
    st.info("先训练得到 models/best_model.pth，再上传一张图片。")
else:
    st.image(uploaded, caption=uploaded.name)
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        # Same path as predict.py: checkpoint preprocess, no augmentation.
        result = predict_image(tmp_path, Path(checkpoint))
        st.success(f"{result['class_name']}  ({result['confidence']:.4f})")
        st.subheader("各类概率")
        st.bar_chart(result["probabilities"])
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
