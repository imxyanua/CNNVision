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


st.set_page_config(page_title="CNNVision", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .stApp { background: #121212; color: #efefe9; }
      .block-container { padding-top: 1rem; max-width: 980px; }
      header[data-testid="stHeader"] { background: #121212; }
      .brand { font-size: 0.95rem; letter-spacing: 0.28em; margin: 0; text-transform: uppercase; }
      .brand-sub { color: #9a9a92; margin: 0.25rem 0 0; font-size: 0.92rem; }
      .stage {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
        min-height: 220px;
        padding: 0.8rem;
      }
      .plate {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        align-items: baseline;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: none;
        padding: 0.85rem 1rem 1rem;
      }
      .plate-class { font-size: 1.8rem; font-weight: 650; margin: 0; }
      .plate-score { color: #b5b5ad; margin: 0; }
      .chips { display: flex; flex-wrap: wrap; gap: 0.45rem; }
      .chip {
        border: 1px solid #3a3a3a;
        padding: 0.2rem 0.55rem;
        font-size: 0.85rem;
        color: #cfcfc7;
      }
      .chip-on { border-color: #efefe9; color: #111; background: #efefe9; }
      div[data-testid="stImage"] img { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state.lang = "vi"
if "checkpoint" not in st.session_state:
    st.session_state.checkpoint = "models/best_model.pth"

top_l, top_r = st.columns([2, 1.4])
with top_r:
    cols = st.columns(3)
    for col, code in zip(cols, LANG_LABELS):
        with col:
            selected = st.session_state.lang == code
            if st.button(
                LANG_LABELS[code],
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state.lang = code
                st.rerun()

lang = st.session_state.lang


def t(key: str) -> str:
    return translate(lang, key)


with top_l:
    st.markdown('<p class="brand">CNNVision</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="brand-sub">{_safe(t("subtitle"))}</p>', unsafe_allow_html=True)

with st.expander(t("settings")):
    st.text_input(t("checkpoint"), key="checkpoint")
    st.caption(t("checkpoint_help"))

uploaded = st.file_uploader(t("upload"), type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"])
checkpoint = Path(st.session_state.checkpoint)

if uploaded is None:
    st.markdown(f'<div class="stage">{_safe(t("stage_empty"))}</div>', unsafe_allow_html=True)
    st.info(t("empty"))
    with st.expander(t("schema")):
        st.markdown(idle_network_svg(), unsafe_allow_html=True)
        st.caption(t("schema_note"))
else:
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        # Same path as predict.py: checkpoint preprocess, no augmentation.
        result = predict_image(tmp_path, checkpoint)
        percent = result["confidence"] * 100
        chips = "".join(
            (
                f'<span class="chip{" chip-on" if name == result["class_name"] else ""}">'
                f"{_safe(name)} {float(value) * 100:.0f}%</span>"
            )
            for name, value in result["probabilities"].items()
        )
        st.markdown('<div class="stage">', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            (
                '<div class="plate">'
                f'<p class="plate-class">{_safe(result["class_name"])}</p>'
                f'<p class="plate-score">{_safe(t("confidence"))} {percent:.1f}%</p>'
                f'<div class="chips">{chips}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        with st.expander(t("schema")):
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
