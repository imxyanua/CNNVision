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
      @keyframes aurora {
        0% { background-position: 0% 40%; }
        100% { background-position: 100% 60%; }
      }
      @keyframes gridShift {
        from { background-position: 0 0, 0 0; }
        to { background-position: 32px 32px, 32px 32px; }
      }
      @keyframes shine {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
      }
      @keyframes glowText {
        0%, 100% { text-shadow: 0 0 8px rgba(125, 255, 224, 0.35); }
        50% { text-shadow: 0 0 26px rgba(125, 255, 224, 0.95); }
      }
      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: none; }
      }
      @keyframes growBar {
        from { transform: scaleX(0); }
        to { transform: scaleX(var(--p, 0)); }
      }
      @keyframes scan {
        0% { top: -20%; }
        100% { top: 120%; }
      }
      .stApp {
        background:
          radial-gradient(900px 420px at 12% -8%, #243868 0%, transparent 55%),
          radial-gradient(700px 380px at 100% 0%, #163f38 0%, transparent 50%),
          linear-gradient(120deg, #070a12, #0c1428, #070a12);
        background-size: 140% 140%;
        animation: aurora 16s ease-in-out infinite alternate;
        color: #e8eefc;
      }
      .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(rgba(90, 150, 255, 0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(90, 150, 255, 0.05) 1px, transparent 1px);
        background-size: 32px 32px;
        animation: gridShift 12s linear infinite;
        z-index: 0;
      }
      .block-container { padding-top: 1.2rem; max-width: 1200px; position: relative; z-index: 1; }
      .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        margin: 0;
        background: linear-gradient(90deg, #7dffe0, #7ec8ff, #b28cff, #7dffe0);
        background-size: 220% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
      }
      .hero-sub { color: #9db0d0; margin: 0.35rem 0 1.1rem; }
      .pred-card {
        border: 1px solid #2a4d7a;
        background: linear-gradient(180deg, #142038 0%, #0c1322 100%);
        border-radius: 18px;
        padding: 1.1rem 1.2rem 0.5rem;
        box-shadow: 0 0 28px rgba(80, 255, 210, 0.12);
        animation: fadeUp 0.45s ease;
        position: relative;
        overflow: hidden;
      }
      .pred-card::after {
        content: "";
        position: absolute;
        left: 0;
        width: 100%;
        height: 28%;
        background: linear-gradient(180deg, transparent, rgba(125, 255, 224, 0.12), transparent);
        animation: scan 2.8s linear infinite;
      }
      .pred-class {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7dffe0;
        line-height: 1.1;
        animation: glowText 2.2s ease-in-out infinite;
        position: relative;
        z-index: 1;
      }
      .pred-score { color: #c5d4f0; font-size: 1.05rem; margin: 0.35rem 0 0.8rem; position: relative; z-index: 1; }
      .prob-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.4rem 0; position: relative; z-index: 1; }
      .prob-name { width: 7rem; color: #c5d4f0; }
      .prob-track {
        flex: 1; height: 10px; background: #18233a; border-radius: 99px; overflow: hidden;
      }
      .prob-fill {
        height: 100%;
        width: 100%;
        transform-origin: left center;
        transform: scaleX(var(--p, 0));
        background: linear-gradient(90deg, #3d7dff, #5cffd0);
        box-shadow: 0 0 12px rgba(92, 255, 208, 0.6);
        animation: growBar 0.8s ease both;
      }
      .prob-val { width: 3.2rem; text-align: right; color: #9db0d0; font-variant-numeric: tabular-nums; }
      div[data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 0 0 1px #2a4d7a, 0 0 24px rgba(80, 160, 255, 0.25);
        animation: fadeUp 0.45s ease;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Checkpoint")
    checkpoint = st.text_input("Path", value="models/best_model.pth", label_visibility="collapsed")
    st.caption("File .pth sau khi train. Class = tên folder dataset, không phát hiện từng vật trong ảnh.")

st.markdown('<p class="hero-title">CNNVISION</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Phân loại cả tấm ảnh thành một class đã train · tín hiệu nơ-ron mang tính minh họa</p>',
    unsafe_allow_html=True,
)

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
                f'<div class="prob-fill" style="--p:{float(value):.4f}"></div>'
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
        st.caption("INPUT → FEATURES → CLASSES. Nốt lớp ra theo Softmax; gạch chạy là hiệu ứng, không phải activation thật trong CNN.")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
