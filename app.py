import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime
import io
from fpdf import FPDF

st.set_page_config(page_title="Skin Cancer AI - Professional", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    .stApp { background: #EEF2F7; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .hero {
        background: linear-gradient(135deg, #050A14 0%, #0F2340 45%, #1A4D7A 100%);
        padding: 32px 24px 28px 24px; border-radius: 24px;
        text-align: center; color: white;
        box-shadow: 0 15px 40px rgba(5,10,20,0.35);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .logo-circle {
        width: 88px; height: 88px; margin: 0 auto;
        background: linear-gradient(135deg, #00E676, #00B0FF);
        border-radius: 50%; display: flex;
        align-items: center; justify-content: center;
        font-size: 44px; box-shadow: 0 0 0 5px rgba(255,255,255,0.12), 0 10px 25px rgba(0,230,118,0.4);
        border: 2px solid white;
    }
    .hero h1 { font-size: 34px; margin: 14px 0 0 0; color: white; letter-spacing: -0.5px; font-weight: 900; }
    .hero p { font-size: 14px; opacity: 0.85; margin: 8px 0 0 0; }
    .badge {
        display: inline-block; background: #00E676; color: #00210A;
        padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 900; margin-top: 16px;
    }
    .badge2 {
        display: inline-block; background: rgba(255,255,255,0.15); color: white;
        padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; margin-top: 16px; margin-left: 6px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .creator {
        margin-top: 18px; font-size: 14px; color: #FFD700;
        background: rgba(255,255,255,0.08);
        display: inline-block; padding: 8px 18px; border-radius: 20px;
        border: 1px solid rgba(255,215,0,0.3);
    }
    .card {
        background: white; padding: 18px 8px 14px 8px; border-radius: 18px; text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.07); border: 1px solid #E8EDF3;
        transition: 0.2s;
    }
    .card-icon {
        width: 44px; height: 44px; margin: 0 auto 8px auto; border-radius: 12px;
        display: flex; align-items: center; justify-content: center; font-size: 22px; color: white;
    }
    .card b { font-size: 14px; display: block; margin-top: 2px; color: #0F172A; }
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
</style>
""", unsafe_allow_html=True)

# --- NEW PROFESSIONAL LOGO + HERO ---
st.markdown("""
<div class="hero">
    <div class="logo-circle">✚</div>
    <h1>Skin Cancer AI</h1>
    <p>Early Detection &nbsp;•&nbsp; Benign vs Malignant &nbsp;•&nbsp; Powered by Deep Learning</p>
    <span class="badge">● AI MODEL ACTIVE</span>
    <span class="badge2">CNN • 128 x 128</span>
    <br>
    <div class="creator">👨‍💻 Developed by <b>Abdul Rehman</b></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- NEW ENGINEERED CARDS ---
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="card"><div class="card-icon" style="background:linear-gradient(135deg,#FF9800,#FF5722);">⚡</div><b>Fast</b><span>2 Sec Result</span></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="card"><div class="card-icon" style="background:linear-gradient(135deg,#2196F3,#0D47A1);">🎯</div><b>Accurate</b><span>CNN Model</span></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="card"><div class="card-icon" style="background:linear-gradient(135deg,#4CAF50,#1B5E20);">🔒</div><b>Private</b><span>No Data Saved</span></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="card"><div class="card-icon" style="background:linear-gradient(135deg,#9C27B0,#6A1B9A);">📄</div><b>PDF Report</b><span>Instant Download</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📤 Upload Skin Lesion Image")
st.caption("Upload a clear, close-up image of the skin lesion for best accuracy.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_cancer_cnn_model.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- PDF GENERATOR FUNCTION ---
def create_pdf_report(image_pil, result, confidence, filename):
    pdf = FPDF()
    pdf.add_page()
    # Header
    pdf.set_fill_color(5, 10, 20)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 10)
    pdf.cell(190, 10, "SKIN CANCER AI", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(10, 23)
    pdf.cell(190, 8, "Early Detection  |  Benign vs Malignant  |  Powered by Deep Learning", align="C")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(255, 215, 0)
    pdf.set_xy(10, 33)
    pdf.cell(190, 8, "Developed by Abdul Rehman", align="C")

    # Image
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    pdf.set_xy(65, 55)
    pdf.image(img_bytes, x=65, y=55, w=80, h=80)

    # Result Box
    y = 145
    if result == "Malignant":
        pdf.set_fill_color(255, 235, 238)
        pdf.set_draw_color(198, 40, 40)
        pdf.set_text_color(198, 40, 40)
        label = "MALIGNANT - Please Consult Doctor Immediately"
    else:
        pdf.set_fill_color(232, 245, 233)
        pdf.set_draw_color(46, 125, 50)
        pdf.set_text_color(46, 125, 50)
        label = "BENIGN - No Cancer Detected"

    pdf.set_xy(15, y)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(180, 16, label, border=1, fill=True, align="C")

    # Details
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "", 12)
    details = [
        ("Date & Time:", datetime.now().strftime('%d-%m-%Y  %H:%M')),
        ("File Name:", filename),
        ("Prediction:", result),
        ("Confidence:", f"{confidence:.2f}%"),
        ("Model:", "CNN (128 x 128)"),
        ("Developed by:", "Abdul Rehman"),
    ]
    y += 28
    for k, v in details:
        pdf.set_xy(20, y)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(45, 8, k)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(120, 8, v)
        y += 10

    # Disclaimer
    y += 10
    pdf.set_xy(15, y)
    pdf.set_fill_color(255, 248, 225)
    pdf.set_draw_color(255, 224, 130)
    pdf.set_text_color(109, 76, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(180, 7, "Disclaimer: This report is for educational purposes only. This is not a final medical diagnosis. Please consult a Dermatologist for confirmation.", border=1, fill=True, align="C")

    pdf.set_xy(10, 275)
    pdf.set_text_color(156, 163, 175)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(190, 8, "Skin Cancer AI  |  Developed by Abdul Rehman  |  Final Year Project", align="C")
    return bytes(pdf.output())

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
        
        # Generate PDF and save in session
        pdf_bytes = create_pdf_report(image, result, confidence, uploaded_file.name)
        st.session_state['pdf_report'] = pdf_bytes

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🔄  Try Another'):
            st.session_state.clear()
            st.rerun()
    with col2:
        if 'pdf_report' in st.session_state:
            st.download_button(label="📥  Download PDF Report", data=st.session_state['pdf_report'], file_name="skin_cancer_report.pdf", mime="application/pdf")
        else:
            st.button("📥  Download PDF Report", disabled=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Please consult a Dermatologist for final diagnosis.</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CA3AF; font-size:12px; margin-top:16px;'>Skin Cancer AI • Developed by <b>Abdul Rehman</b> • Final Year Project • Powered by CNN</p>", unsafe_allow_html=True)
