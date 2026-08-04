"""Interactive land-use classification and temporal change-detection dashboard."""
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image
from models.resnet import ResNet18LandUse
from utils.data import eval_transform
from utils.temporal import compare_images

st.set_page_config(page_title="Satellite Land-Use Classifier", layout="wide")
@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=False); classes = state.get("classes", [str(i) for i in range(10)])
    model = ResNet18LandUse(len(classes), pretrained=False).to(device); model.load_state_dict(state.get("model_state", state)); model.eval()
    return model, classes, device
st.title("Satellite Land-Use Classification & Change Detection")
checkpoint = st.sidebar.text_input("Model checkpoint", "checkpoints/best_model.pt")
if not Path(checkpoint).exists(): st.warning("Train the model first (`python train.py`) or choose a valid checkpoint."); st.stop()
model, classes, device = load_model(checkpoint); transform = eval_transform()
classify_tab, change_tab = st.tabs(["Land-use classification", "Temporal change detection"])
with classify_tab:
    upload = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"], key="classify")
    if upload:
        image = Image.open(upload).convert("RGB")
        with torch.inference_mode(): probabilities = torch.softmax(model(transform(image).unsqueeze(0).to(device)), 1)[0].cpu()
        top = probabilities.topk(min(3, len(classes))); st.image(image, caption="Input image", width=360); st.subheader(f"Prediction: {classes[top.indices[0]]}")
        st.bar_chart({classes[i]: float(probabilities[i]) for i in top.indices})
with change_tab:
    before_file = st.file_uploader("Earlier image", type=["png", "jpg", "jpeg"], key="before")
    after_file = st.file_uploader("Later image", type=["png", "jpg", "jpeg"], key="after")
    threshold = st.slider("Change alert threshold (similarity below)", 0.0, 1.0, .85)
    if before_file and after_file:
        before, after = Image.open(before_file), Image.open(after_file); similarity, heatmap = compare_images(model, before, after, transform, device)
        st.metric("Cosine similarity", f"{similarity:.3f}", "Change detected" if similarity < threshold else "No major change")
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(before); axes[0].set_title("Earlier"); axes[0].axis("off")
        axes[1].imshow(after); axes[1].set_title("Later"); axes[1].axis("off")
        plotted = axes[2].imshow(heatmap, cmap="hot", vmin=0, vmax=1); axes[2].set_title("Embedding change heatmap"); axes[2].axis("off")
        figure.colorbar(plotted, ax=axes[2], fraction=.046); st.pyplot(figure)
