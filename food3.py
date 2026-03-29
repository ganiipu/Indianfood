import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO

st.set_page_config(page_title="Indian Food 🚀", page_icon="🍛")
st.title("Indian Food Image Classifier")

@st.cache_resource
def load_model(model_id="ganiipu/food"):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    return processor, model

processor, model = load_model()

st.subheader("Upload an image or provide a URL")

uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or enter an image URL:")

run = st.button("Run Prediction")
clear = st.button("Clear")

if clear:
    st.rerun()

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif image_url:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, timeout=15, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image from URL: {e}")

if run:
    if image is not None:
        st.image(image, caption="Selected Image", width="stretch")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = probs.argmax().item()
        predicted_label = model.config.id2label[predicted_idx]
        confidence = probs[0, predicted_idx].item()

        st.success(f"Prediction: {predicted_label} ({confidence*100:.2f}% confidence)")
    else:
        st.warning("Please upload an image or provide a URL before submitting.")