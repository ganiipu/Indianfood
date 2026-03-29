import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO

# ---------- Page Config ----------
st.set_page_config(page_title="Indian Food 🚀", page_icon="🍛")
st.title("Indian Food Image Classifier")

# ---------- CSS for 3D Buttons & URL Box ----------
st.markdown("""
<style>
/* URL input outline highlight */
.stTextInput>div>input {
    border: 2px solid #FF9800 !important;
    border-radius: 8px;
    padding: 0.5em;
    font-size: 1rem;
}

/* Buttons container with space between */
.button-row {
    display: flex;
    justify-content: space-between;
    margin-top: 0.3em;
}

/* 3D buttons */
.button-3d-submit {
    background: linear-gradient(to bottom, #4CAF50, #388E3C);
    border-radius: 12px;
    border: none;
    color: white;
    font-weight: bold;
    padding: 0.5em 1em;
    font-size: 0.9rem;
    box-shadow: 0 4px #2E7D32;
    cursor: pointer;
    transition: 0.1s;
}
.button-3d-submit:active {
    box-shadow: 0 2px #2E7D32;
    transform: translateY(3px);
}

/* Clear button red */
.button-3d-clear {
    background: linear-gradient(to bottom, #F44336, #D32F2F);
    border-radius: 12px;
    border: none;
    color: white;
    font-weight: bold;
    padding: 0.5em 1em;
    font-size: 0.9rem;
    box-shadow: 0 4px #B71C1C;
    cursor: pointer;
    transition: 0.1s;
}
.button-3d-clear:active {
    box-shadow: 0 2px #B71C1C;
    transform: translateY(3px);
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_model(model_id="ganiipu/food"):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    return processor, model

processor, model = load_model()

# ---------- User Input ----------
st.subheader("Upload an image or provide a URL")

uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])
image_url = None

# Conditional URL input if no file uploaded
if uploaded_file is None:
    image_url = st.text_input("Or enter an image URL:")

    # Buttons row under URL box
    st.markdown(
        f"""
        <div class="button-row">
            <button class="button-3d-submit" onclick="document.querySelector('button[kind=primary]').click();">Submit</button>
            <button class="button-3d-clear" onclick="window.location.reload();">Clear</button>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Streamlit Submit Button ----------
# We still need a st.button to trigger processing
submit_clicked = st.button("Run Prediction", key="submit_hidden", help="Hidden trigger for Submit")

# ---------- Load Image ----------
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image from URL: {e}")

# ---------- Run Inference ----------
if submit_clicked:
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = probs.argmax().item()
        predicted_label = model.config.id2label[predicted_idx]
        confidence = probs[0, predicted_idx].item()

        st.success(f"Prediction: **{predicted_label}** ({confidence*100:.2f}% confidence)")
    else:
        st.warning("Please upload an image or provide a URL before submitting!")