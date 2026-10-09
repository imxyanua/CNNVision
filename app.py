from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app_i18n import LANG_LABELS, translate
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


st.set_page_config(page_title="CNNVision", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .stApp { background: #f4f1ea; color: #1c1c1c; }
      .block-container { padding-top: 1.4rem; max-width: 1080px; }
      h1, h2, h3 { font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif; }
      .top-name { font-size: 1.7rem; font-weight: 700; margin: 0; letter-spacing: 0.02em; }
      .top-sub { color: #5a564e; margin: 0.2rem 0 0; }
      .panel {
        background: #fffdf8;
        border: 1px solid #d8d2c6;
        padding: 1rem 1.1rem;
      }
      .pred-class { font-size: 2rem; font-weight: 700; margin: 0; }
      .pred-score { color: #5a564e; margin: 0.3rem 0 0.8rem; }
      .prob-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.35rem 0; }
      .prob-name { width: 7rem; }
      .prob-track { flex: 1; height: 8px; background: #ece7dc; overflow: hidden; }
      .prob-fill { height: 100%; width: 100%; transform-origin: left center; transform: scaleX(var(--p, 0)); background: #1f4d8f; }
      .prob-val { width: 3.4rem; text-align: right; color: #5a564e; font-variant-numeric: tabular-nums; }
      div[data-testid="stImage"] img { border: 1px solid #d8d2c6; background: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state.lang = "vi"

header_l, header_r = st.columns([2.4, 1.2])
with header_r:
    st.caption(translate(st.session_state.lang, "language"))
    st.radio(
        translate(st.session_state.lang, "language"),
        options=list(LANG_LABELS),
        format_func=lambda code: LANG_LABELS[code],
        horizontal=True,
        label_visibility="collapsed",
        key="lang",
    )
lang = st.session_state.lang

def t(key: str) -> str:
    return translate(lang, key)

with header_l:
    st.markdown('<p class="top-name">CNNVision</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="top-sub">{_safe(t("subtitle"))}</p>', unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.header(t("checkpoint"))
    checkpoint = st.text_input("Path", value="models/best_model.pth", label_visibility="collapsed")
    st.caption(t("checkpoint_help"))

uploaded = st.file_uploader(t("upload"), type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"])

if uploaded is None:
    st.info(t("empty"))
    st.markdown(t("schema"))
    st.markdown(idle_network_svg(), unsafe_allow_html=True)
    st.caption(t("schema_note"))
else:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(t("image"))
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
            st.markdown(t("result"))
            st.markdown(
                (
                    '<div class="panel">'
                    f'<p class="pred-class">{_safe(result["class_name"])}</p>'
                    f'<p class="pred-score">{_safe(t("confidence"))}: {percent:.1f}%</p>'
                    f"{bars}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        st.markdown(t("schema"))
        st.markdown(
            probability_network_svg(result["probabilities"], result["class_name"]),
            unsafe_allow_html=True,
        )
        st.caption(t("schema_note"))
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
