"""Streamlit app: upload a brain MRI slice and get the predicted tumour type with confidence scores.

Run from the repository root:   streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from model_utils import CLASSES, grad_cam, load_bundle, overlay, predict_proba, preprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pt"
INFO_PATH = ROOT / "models" / "model_info.json"
SAMPLES = sorted((ROOT / "sample_images").glob("*.jpg"))

DESCRIPTIONS = {
    "glioma": "A tumour that grows from the glial (supporting) cells of the brain itself.",
    "meningioma": "A tumour of the meninges, the membranes that cover the brain; often slow-growing.",
    "no_tumor": "No tumour pattern was recognised in this scan.",
    "pituitary": "A tumour of the pituitary gland at the base of the brain.",
}
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    model, info = load_bundle(MODEL_PATH)
    return model, info


st.title("🧠 Brain Tumor MRI Classifier")
st.warning("Educational / research prototype. It is **not** a medical device and must not be used for diagnosis or "
           "treatment decisions. A qualified radiologist must review every scan.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}. Run the notebook (Stage 4) to create it.")
    st.stop()

model, info = get_model()

with st.sidebar:
    st.header("About the model")
    st.write(f"**Architecture:** `{info['arch']}`")
    st.write(f"**Input size:** {info['img_size']} x {info['img_size']}  ({info.get('preprocess', 'plain')} pre-processing)")
    st.write("**Classes:** " + ", ".join(CLASSES))
    if INFO_PATH.exists():
        metrics = json.loads(INFO_PATH.read_text()).get("test_metrics", {})
        if metrics:
            st.write("**Held-out test set:**")
            st.table(pd.DataFrame({"score": {k.replace("test_", ""): round(v, 3) for k, v in metrics.items()}}))
    st.caption("Trained on one public dataset (2,443 slices). Results on scans from other scanners or hospitals "
               "may be worse.")
    show_cam = st.checkbox("Show Grad-CAM (where the model looked)", value=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Choose a scan")
    upload = st.file_uploader("Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    sample = st.selectbox("...or try a sample scan", ["(none)"] + [p.name for p in SAMPLES])

pil_image = None
if upload is not None:
    try:
        pil_image = Image.open(upload)
        pil_image.load()
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please upload a JPG or PNG.")
        st.stop()
elif sample != "(none)":
    pil_image = Image.open(ROOT / "sample_images" / sample)

if pil_image is None:
    with right:
        st.info("Upload a scan or pick a sample on the left to get a prediction.")
    st.stop()

x, resized = preprocess(pil_image, info)
probs = predict_proba(model, x)
top = int(probs.argmax())
confidence = float(probs[top])

with left:
    st.image(pil_image, caption="Input scan", width="stretch")

with right:
    st.subheader("2. Prediction")
    st.metric("Predicted class", CLASSES[top].replace("_", " ").title(), f"{confidence:.1%} confidence")
    st.write(DESCRIPTIONS[CLASSES[top]])
    if confidence < LOW_CONFIDENCE:
        st.warning("Low confidence: the model is unsure about this scan. Treat the result with extra caution.")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES), horizontal=True)

    if show_cam:
        st.subheader("3. Where the model looked (Grad-CAM)")
        cam = grad_cam(model, info["arch"], x.clone(), class_idx=top)
        st.image(overlay(resized, cam), caption="Red = regions that most increased the predicted class score",
                 width="stretch")
        st.caption("Heat-maps show what influenced the model, not where a tumour is proven to be. If the red area lies on the "
                   "black background instead of the brain, the prediction should not be trusted.")
