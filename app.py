import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Skin Cancer Classification", layout="centered")

# --- STYLISH BUTTONS KA CSS ---
st.markdown("""
<style>
    /* Predict Button - Bara aur Green */
    div[data-testid="stButton"] button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        padding: 12px;
        border: none;
        transition: 0.3s;
    }
    /* Download Button - Neela */
    div[data-testid="stDownloadButton"] button {
        width: 100%;
        border-radius: 12px;
        background: #2196F3;
        color: white;
        font-weight: 700;
        font-size: 15px;
        padding: 12px;
        border: none;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: #1976D2;
        color: white;
    }
    /* Clear Button ko halka Grey */
    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Skin Cancer (Benign vs Malignant) Classifier")
st.write("Upload an image of the skin lesion, and the trained CNN model will predict whether it is Benign or Malignant.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_cancer_cnn_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    resized_image = image.resize((128, 128))
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
        
    # --- BARA PREDICT BUTTON ---
    if st.button('🔍  PREDICT NOW', type="primary"):
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)[0][0]
            
        if prediction > 0.5:
            result = "Malignant"
            confidence = prediction * 100
            st.error(f"**Prediction: Malignant** (Confidence: {confidence:.2f}%)")
        else:
            result = "Benign"
            confidence = (1 - prediction) * 100
            st.success(f"**Prediction: Benign** (Confidence: {confidence:.2f}%)")
        
        report_text = f"""SKIN CANCER CLASSIFICATION REPORT

Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}
File Name: {uploaded_file.name}
Prediction: {result}
Confidence: {confidence:.2f}%
Model: CNN (128x128)

Note: This is an AI-based prediction for educational purpose only. Please consult a dermatologist for final diagnosis.
"""
        st.session_state['report'] = report_text

    st.markdown("---")

    # --- NEECHE 2 CHOTE BUTTON SAATH SAATH ---
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🔄  Try Another'):
            st.session_state.clear()
            st.rerun()

    with col2:
        if 'report' in st.session_state:
            st.download_button(
                label="📥  Download Report",
                data=st.session_state['report'],
                file_name="skin_cancer_report.txt",
                mime="text/plain"
            )
        else:
            st.button("📥  Download Report", disabled=True)
