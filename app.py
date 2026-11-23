import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Skin Cancer AI - Professional", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    .stApp { background: #EEF2F7; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .hero {
        background: linear-gradient(135deg, #050A14 0%, #0F2340 45%, #1A4D7A 100%);
        padding: 36px 24px 28px 24px; border-radius: 22px;
        text-align: center; color: white;
        box-shadow: 0 15px 40px rgba(5,10,20,0.35);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero h1 { font-size: 32px; margin: 6px 0 0 0; color: white; letter-spacing: -0.5px; }
    .hero p { font-size: 14px; opacity: 0.85; margin: 10px 0 0 0; }
    .badge {
        display: inline-block; background: #00E676; color: #00210A;
        padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 900; margin-top: 16px;
    }
    .badge2 {
        display: inline-block; background: rgba(255,255,255,0.15); color: white;
        padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; margin-top: 16px; margin-left: 6px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .card {
        background: white; padding: 16px 10px; border-radius: 16px; text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06); border: 1px solid #E8EDF3;
    }
    .card b { font-size: 14px; display: block; margin-top: 4px; }
    .card span { font-size: 11px; color: #6B7280; }
    div.stButton > button {
        width: 100% !important; border-radius: 14px !important;
        padding: 15px !important; font-weight: 800 !important; font-size: 16px !important;
        background: linear-gradient(90deg, #FF3B30, #FF6B00) !important;
        color: white !important; border: none !important;
        box-shadow: 0 8px 20px rgba(255,59,48,0.30) !important;
    }
    div.stDownloadButton > button {
        width: 100% !important; border-radius: 14px !important; padding: 15px !important;
        font-weight: 800 !important; font-size: 15px !important;
        background: #0F172A !important; color: white !important; border: none !important;
    }
    .result-benign { background: #E8F5E9; border: 2px solid #2E7D32; padding: 20px; border-radius: 16px; text-align: center; }
    .result-malignant { background: #FFEBEE; border: 2px solid #C62828; padding: 20px; border-radius: 16px; text-align: center; }
    .disclaimer { background: #FFF8E1; padding: 14px; border-radius: 14px; font-size: 12px; color: #6D4C00; border: 1px solid #FFE082; }
    .upload-box { background: white; padding: 16px; border-radius: 16px; border: 2px dashed #CBD5E1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div style="font-size:48px;">🩺</div>
    <h1>Skin Cancer AI</h1>
    <p>Early Detection &nbsp;•&nbsp; Benign vs Malignant &nbsp;•&nbsp; Powered by Deep Learning</p>
    <span class="badge">● AI MODEL ACTIVE</span>
    <span class="badge2">CNN • 128 x 128</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="card">⚡<b>Fast</b><span>2 Sec Result</span></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="card">🎯<b>Accurate</b><span>CNN Model</span></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="card">🔒<b>Private</b><span>No Data Saved</span></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="card">📄<b>Report</b><span>PDF Ready</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📤 Upload Skin Lesion Image")
st.caption("صاف اور قریب سے لی ہوئی تصویر اپلوڈ کریں، بہترین رزلٹ کے لیے۔")

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
        with st.spinner('AI Analyzing...'):
            prediction = model.predict(img_array)[0][0]
            
        if prediction > 0.5:
            result = "Malignant"
            confidence = prediction * 100
            st.markdown(f'<div class="result-malignant"><h2 style="color:#C62828; margin:0;">⚠️ MALIGNANT</h2><p style="margin:6px 0 0 0;">Confidence: <b>{confidence:.2f}%</b> • Please Consult Doctor Immediately</p></div>', unsafe_allow_html=True)
        else:
            result = "Benign"
            confidence = (1 - prediction) * 100
            st.markdown(f'<div class="result-benign"><h2 style="color:#2E7D32; margin:0;">✅ BENIGN</h2><p style="margin:6px 0 0 0;">Confidence: <b>{confidence:.2f}%</b> • No Cancer Detected</p></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"Confidence: {confidence:.1f}%")
        st.progress(int(confidence))
        
        report_text = f"""SKIN CANCER AI - CLASSIFICATION REPORT

Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}
File: {uploaded_file.name}
Prediction: {result}
Confidence: {confidence:.2f}%
Model: CNN (128x128)
Note: Educational purpose only. Consult Dermatologist.
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
st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> یہ ٹول صرف تعلیمی مقصد کے لیے ہے۔ فائنل تشخیص کے لیے ہمیشہ ڈرماٹولوجسٹ سے رجوع کریں۔</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CA3AF; font-size:12px; margin-top:16px;'>Skin Cancer AI • Final Year Project • Made with ❤️ using CNN</p>", unsafe_allow_html=True)
