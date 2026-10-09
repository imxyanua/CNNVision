from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from inference.predict import predict_image
from visualization.neurons import idle_network_svg, probability_network_svg


def _safe(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


st.set_page_config(page_title="CNNVision", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(900px 420px at 12% -8%, #1b2c55 0%, transparent 55%),
          radial-gradient(700px 380px at 100% 0%, #12352f 0%, transparent 50%),
          #070a12;
        color: #e8eefc;
      }
      .block-container { padding-top: 1.4rem; max-width: 1200px; }
      h1 { letter-spacing: 0.06em; }
      .hero-sub { color: #9db0d0; margin-top: -0.6rem; margin-bottom: 1.2rem; }
      .pred-card {
        border: 1px solid #1e3358;
        background: linear-gradient(180deg, #10182b 0%, #0c1322 100%);
        border-radius: 18px;
        padding: 1.1rem 1.2rem 0.4rem;
      }
      .pred-class {
        font-size: 2.1rem;
        font-weight: 700;
        color: #7dffe0;
        line-height: 1.1;
      }
      .pred-score { color: #c5d4f0; font-size: 1.05rem; margin: 0.35rem 0 0.8rem; }
      .prob-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.35rem 0; }
      .prob-name { width: 7rem; color: #c5d4f0; }
      .prob-track {
        flex: 1; height: 9px; background: #18233a; border-radius: 99px; overflow: hidden;
      }
      .prob-fill { height: 100%; background: linear-gradient(90deg, #3d7dff, #5cffd0); }
      .prob-val { width: 3.2rem; text-align: right; color: #9db0d0; font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Checkpoint")
    checkpoint = st.text_input("Path", value="models/best_model.pth", label_visibility="collapsed")
    st.caption("File .pth sau khi train. Class = tên folder dataset, không phát hiện từng vật trong ảnh.")

st.title("CNNVision")
st.markdown('<p class="hero-sub">Phân loại cả tấm ảnh thành một class đã train · sơ đồ nơ-ron mang tính minh họa</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Kéo ảnh vào đây",
    type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"],
)

if uploaded is None:
    st.info("Cần `models/best_model.pth` rồi upload một ảnh. Kết quả là một nhãn cho cả tấm, không khoanh từng vật.")
    st.markdown(idle_network_svg(), unsafe_allow_html=True)
else:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

    suffix = Path(uploaded.name).suffix.lower() or ".png"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        # Same path as predict.py: checkpoint preprocess, no augmentation.
        result = predict_image(tmp_path, Path(checkpoint))
        percent = result["confidence"] * 100
        bars = "".join(
            (
                '<div class="prob-row">'
                f'<div class="prob-name">{_safe(name)}</div>'
                '<div class="prob-track">'
                f'<div class="prob-fill" style="width:{float(value) * 100:.1f}%"></div>'
                "</div>"
                f'<div class="prob-val">{float(value) * 100:.1f}%</div>'
                "</div>"
            )
            for name, value in result["probabilities"].items()
        )
        with right:
            st.markdown(
                (
                    '<div class="pred-card">'
                    f'<div class="pred-class">{_safe(result["class_name"])}</div>'
                    f'<div class="pred-score">Độ tin cậy {percent:.1f}%</div>'
                    f"{bars}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            probability_network_svg(result["probabilities"], result["class_name"]),
            unsafe_allow_html=True,
        )
        st.caption("INPUT → FEATURES → CLASSES. Kích thước nốt lớp ra theo xác suất Softmax, không phải activation thật trong CNN.")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
