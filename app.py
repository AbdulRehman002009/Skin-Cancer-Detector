import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Skin Cancer AI | Professional Detection", page_icon="🩺", layout="centered")

# --- ULTRA MODERN PROFESSIONAL CSS ---
st.markdown("""
<style>
    /* Background */
    .stApp { background: #F6F8FB; }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top Card */
    .hero {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 55%, #2C5364 100%);
        padding: 32px 24px 26px 24px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(15,32,39,0.25);
    }
    .hero h1 { font-size: 30px; margin: 0; color: white; letter-spacing: -0.5px; }
    .hero p { font-size: 14px; opacity: 0.85; margin: 10px 0 0 0; }
    .badge {
        display: inline-block; background: #00C853; color: white;
        padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 800; margin-top: 14px;
    }
    /* 3 Small Cards */
    .info-card {
        background: white; padding: 14px 10px; border-radius: 14px;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #EEF1F5;
    }
    .info-card b { font-size: 13px; display: block; }
    .info-card span { font-size: 11px; color: #6B7280; }

    /* Buttons */
    div.stButton > button {
        width: 100% !important; border-radius: 12px !important;
        padding: 14px !important; font-weight: 800 !important; font-size: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10) !important;
    }
    div.stDownloadButton > button {
        width: 100% !important; border-radius: 12px !important; padding: 14px !important;
        font-weight: 800 !important; font-size: 15px !important;
        background: #111827 !important; color: white !important; border: none !important;
    }
    /* Result Box */
    .result-benign { background: #E8F5E9; border: 1px solid #A5D6A7; padding: 18px; border-radius: 14px; text-align: center; }
    .result-malignant { background: #FFEBEE; border: 1px solid #EF9A9A; padding: 18px; border-radius: 14px; text-align: center; }
    .disclaimer { background: #FFF8E1; padding: 12px 14px; border-radius: 12px; font-size: 12px; color: #6D4C00; border: 1px solid #FFE082; }
</style>
""", unsafe_allow_html=True)

# --- HERO ---
st.markdown("""
<div class="hero">
    <div style="font-size:42px;">🩺</div>
    <h1>Skin Cancer Classifier</h1>
    <p>AI-Powered Early Detection &nbsp;•&nbsp; Benign vs Malignant</p>
    <span class="badge">● AI MODEL ACTIVE &nbsp;•&nbsp; CNN</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 3 CARDS ---
c1, c2, c3 = st.columns(3)
with c1: st.markdown('<div class="info-card"><b>⚡ Fast</b><span>Result in 2 sec</span></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="info-card"><b>🎯 Accurate</b><span>CNN Deep Learning</span></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="info-card"><b>🔒 Private</b><span>Image not saved</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📤 Upload Skin Image")
st.caption("Clear, close-up image upload karein. Best result ke liye lesion saaf nazar aana chahiye.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_cancer_cnn_model.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    resized_image = image.resize((128, 128))
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
        
    if st.button('🔍  ANALYZE NOW', type="primary"):
        with st.spinner('AI Analyzing Image...'):
            prediction = model.predict(img_array)[0][0]
            
        if prediction > 0.5:
            result = "Malignant"
            confidence = prediction * 100
            st.markdown(f'<div class="result-malignant"><h2 style="color:#C62828; margin:0;">⚠️ Malignant</h2><p style="margin:5px 0 0 0;">Confidence: <b>{confidence:.2f}%</b></p></div>', unsafe_allow_html=True)
        else:
            result = "Benign"
            confidence = (1 - prediction) * 100
            st.markdown(f'<div class="result-benign"><h2 style="color:#2E7D32; margin:0;">✅ Benign</h2><p style="margin:5px 0 0 0;">Confidence: <b>{confidence:.2f}%</b></p></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Confidence Level")
        st.progress(int(confidence))
        
        report_text = f"""SKIN CANCER CLASSIFICATION REPORT

Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}
File Name: {uploaded_file.name}
Prediction: {result}
Confidence: {confidence:.2f}%
Model: CNN (128x128)
Note: AI prediction for educational purpose only. Consult dermatologist for final diagnosis.
"""
        st.session_state['report'] = report_text

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🔄  Try Another'):
            st.session_state.clear()
            st.rerun()
    with col2:
        if 'report' in st.session_state:
            st.download_button(label="📥  Download Report", data=st.session_state['report'], file_name="skin_cancer_report.txt", mime="text/plain")
        else:
            st.button("📥  Download Report", disabled=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> Ye tool sirf educational purpose ke liye hai. Final diagnosis ke liye hamesha Dermatologist se consult karein.</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CA3AF; font-size:12px; margin-top:18px;'>Skin Cancer AI • Final Year Project • Powered by CNN</p>", unsafe_allow_html=True)
