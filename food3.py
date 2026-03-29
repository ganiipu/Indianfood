import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO

st.set_page_config(page_title="Indian Food 🚀", page_icon="🍛", layout="centered")
st.title("Indian Food Image Classifier")
st.write("Upload an image or paste an image URL, then click Predict.")

st.markdown(
    """
    <style>
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_ID = "ganiipu/food"

@st.cache_resource
def load_model(model_id=MODEL_ID):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    return processor, model

try:
    processor, model = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

st.subheader("Input")
uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"], key="uploader")
image_url = st.text_input("Or enter an image URL", key="image_url")

col1, col2 = st.columns(2)
with col1:
    predict = st.button("Predict", type="primary")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state["uploader"] = None
    st.session_state["image_url"] = ""
    st.rerun()

image = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

elif image_url.strip():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url.strip(), timeout=15, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image from URL: {e}")

if predict:
    if image is None:
        st.warning("Please upload an image or provide a valid image URL.")
    else:
        st.image(image, caption="Selected Image", width="stretch")

        try:
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_idx = int(probs.argmax().item())
            predicted_label = model.config.id2label.get(predicted_idx, str(predicted_idx))
            confidence = float(probs[0, predicted_idx].item())

            st.success(f"Prediction: {predicted_label} ({confidence*100:.2f}% confidence)")
        except Exception as e:
            st.error(f"Inference failed: {e}")