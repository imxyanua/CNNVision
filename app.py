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
      @keyframes driftA {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(10%, 14%) scale(1.18); }
        100% { transform: translate(-8%, 6%) scale(1); }
      }
      @keyframes driftB {
        0% { transform: translate(0, 0) scale(1.05); }
        50% { transform: translate(-12%, -8%) scale(1.2); }
        100% { transform: translate(8%, 10%) scale(1); }
      }
      @keyframes driftC {
        0% { transform: translate(0, 0); }
        50% { transform: translate(8%, -12%); }
        100% { transform: translate(-10%, 8%); }
      }
      .stApp { background: #10131a; color: #ecece6; }
      .block-container { padding-top: 0.8rem; max-width: 1400px; }
      header[data-testid="stHeader"] { background: transparent; }
      .orbs span {
        position: fixed;
        border-radius: 50%;
        filter: blur(70px);
        opacity: 0.5;
        pointer-events: none;
        z-index: 0;
      }
      .orbs span:nth-child(1) {
        width: 460px; height: 460px; background: #2b3f73; top: -120px; left: -80px;
        animation: driftA 16s ease-in-out infinite;
      }
      .orbs span:nth-child(2) {
        width: 380px; height: 380px; background: #1e4a3f; right: -60px; top: 12%;
        animation: driftB 18s ease-in-out infinite;
      }
      .orbs span:nth-child(3) {
        width: 320px; height: 320px; background: #4a2e4a; bottom: -80px; left: 28%;
        animation: driftC 20s ease-in-out infinite;
      }
      .block-container { position: relative; z-index: 1; }
      .brand { font-size: 1.05rem; letter-spacing: 0.28em; margin: 0; text-transform: uppercase; }
      .brand-sub { color: #b3b3aa; margin: 0.3rem 0 0.8rem; }
      .lang-label {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin: 0 0 0.45rem;
      }
      .panel {
        background: rgba(18, 20, 28, 0.72);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.85rem 0.95rem 1rem;
        backdrop-filter: blur(10px);
        margin-bottom: 0.8rem;
      }
      .panel h3 { margin: 0 0 0.55rem; font-size: 0.95rem; letter-spacing: 0.08em; text-transform: uppercase; }
      .stage {
        background: rgba(8, 8, 10, 0.65);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        min-height: 240px;
        padding: 0.7rem;
      }
      .plate {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem 1.1rem;
        align-items: baseline;
        margin-top: 0.7rem;
      }
      .plate-class { font-size: 1.7rem; font-weight: 700; margin: 0; }
      .plate-score { color: #c5c5bc; margin: 0; }
      .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
      .chip {
        border: 1px solid rgba(255,255,255,0.16);
        padding: 0.18rem 0.5rem;
        font-size: 0.82rem;
        border-radius: 999px;
      }
      .chip-on { background: #efefe9; color: #111; border-color: #efefe9; }
      div[data-testid="stImage"] img { width: 100%; border-radius: 8px; }
    </style>
    <div class="orbs"><span></span><span></span><span></span></div>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state.lang = "vi"
if "checkpoint" not in st.session_state:
    st.session_state.checkpoint = "models/best_model.pth"

st.markdown('<p class="brand">CNNVision</p>', unsafe_allow_html=True)

lang = st.session_state.lang


def t(key: str) -> str:
    return translate(lang, key)


st.markdown(f'<p class="brand-sub">{_safe(t("subtitle"))}</p>', unsafe_allow_html=True)

st.markdown(f'<p class="lang-label">{_safe(t("lang_bar"))}</p>', unsafe_allow_html=True)
lang_cols = st.columns(3)
for col, code in zip(lang_cols, LANG_LABELS):
    with col:
        selected = st.session_state.lang == code
        if st.button(
            LANG_LABELS[code],
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            st.session_state.lang = code
            st.rerun()

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown(f'<div class="panel"><h3>{_safe(t("settings"))}</h3></div>', unsafe_allow_html=True)
    st.text_input(t("checkpoint"), key="checkpoint")
    st.caption(t("checkpoint_help"))
    uploaded = st.file_uploader(t("upload"), type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"])
    checkpoint = Path(st.session_state.checkpoint)

    result = None
    if uploaded is None:
        st.markdown(f'<div class="stage">{_safe(t("stage_empty"))}<br>{_safe(t("empty"))}</div>', unsafe_allow_html=True)
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
            st.image(uploaded, use_container_width=True)
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
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

with right:
    st.markdown(f'<div class="panel"><h3>{_safe(t("schema"))}</h3></div>', unsafe_allow_html=True)
    if result is None:
        st.markdown(idle_network_svg(), unsafe_allow_html=True)
    else:
        st.markdown(
            probability_network_svg(result["probabilities"], result["class_name"]),
            unsafe_allow_html=True,
        )
    st.caption(t("schema_note"))
