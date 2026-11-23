import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime
import io
from fpdf import FPDF

st.set_page_config(page_title="SkinSense AI - Abdul Rehman", page_icon="✦", layout="centered")

st.markdown("""
<style>
    .stApp { background: #F0F4F8; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #042F2E 0%, #0F766E 45%, #14B8A6 100%);
        padding: 30px 20px 24px 20px; border-radius: 24px;
        text-align: center; color: white;
        box-shadow: 0 15px 35px rgba(4,47,46,0.35);
        border: 1px solid rgba(255,255,255,0.15);
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ""; position: absolute; top: -60px; right: -60px;
        width: 180px; height: 180px; background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    /* APNA LOGO - AR MONOGRAM */
    .my-logo {
        width: 82px; height: 82px; margin: 0 auto; position: relative; z-index: 2;
        background: linear-gradient(135deg, #F59E0B, #FDE68A);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 32px; font-weight: 900; color: #042F2E;
        border: 3px solid white;
        box-shadow: 0 10px 25px rgba(245,158,11,0.5);
        letter-spacing: -1px;
    }
    .my-logo span { font-size: 14px; position: absolute; top: -6px; right: -2px; background: #fff; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #0F766E; font-size: 16px; font-weight: 900; }
    .hero h1 { font-size: 34px; margin: 14px 0 0 0; color: white; font-weight: 900; letter-spacing: -0.5px; position: relative; z-index: 2; }
    .hero h1 span { color: #FDE68A; }
    .hero p { font-size: 13px; opacity: 0.9; margin: 6px 0 0 0; position: relative; z-index: 2; }
    .badge-green { display: inline-block; background: #CCFBF1; color: #134E4A; padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 800; margin-top: 14px; position: relative; z-index: 2; }
    .badge-white { display: inline-block; background: rgba(255,255,255,0.18); color: white; padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; margin-top: 14px; margin-left: 6px; border: 1px solid rgba(255,255,255,0.3); position: relative; z-index: 2; }
    .creator-badge {
        background: linear-gradient(90deg, #F59E0B, #F97316); color: white; font-weight: 800; font-size: 13px;
        padding: 9px 22px; border-radius: 20px; display: inline-block; margin-top: 16px;
        box-shadow: 0 6px 18px rgba(249,115,22,0.4); position: relative; z-index: 2;
    }
    .card { background: white; padding: 16px 6px 12px 6px; border-radius: 16px; text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.06); border: 1px solid #E5E7EB; }
    .card-icon { font-size: 26px; margin-bottom: 4px; }
    .card b { font-size: 13px; display: block; color: #111827; }
    .card span { font-size: 10.5px; color: #6B7280; display: block; margin-top: 2px; }
    div.stButton > button { width: 100% !important; border-radius: 12px !important; padding: 13px !important; font-weight: 800 !important; font-size: 15px !important; background: linear-gradient(90deg, #0F766E, #14B8A6) !important; color: white !important; border: none !important; }
    div.stDownloadButton > button { width: 100% !important; border-radius: 12px !important; padding: 13px !important; font-weight: 700 !important; font-size: 14px !important; background: white !important; color: #0F766E !important; border: 2px solid #14B8A6 !important; }
    .result-benign { background: #ECFDF5; border: 1px solid #6EE7B7; padding: 16px; border-radius: 14px; }
    .result-malignant { background: #FEF2F2; border: 1px solid #FCA5A5; padding: 16px; border-radius: 14px; }
    .disclaimer { background: #FFFBEB; padding: 12px; border-radius: 12px; font-size: 11px; color: #92400E; border: 1px solid #FDE68A; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- HERO WITH APNA LOGO ---
st.markdown("""
<div class="hero">
    <div class="my-logo">AR<span>+</span></div>
    <h1>Skin<span>Sense</span> AI</h1>
    <p>Early Skin Cancer Detection • Benign vs Malignant • Powered by Deep Learning</p>
    <span class="badge-green">● AI MODEL ACTIVE</span>
    <span class="badge-white">CNN • 128 x 128 • High Accuracy</span>
    <br>
    <div class="creator-badge">✦ Developed by Abdul Rehman ✦</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="card"><div class="card-icon">🔍</div><b>Early Detection</b><span>Detect Skin Issues Early</span></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="card"><div class="card-icon">🧠</div><b>Smart AI</b><span>Deep Learning CNN</span></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="card"><div class="card-icon">🔒</div><b>Safe & Private</b><span>Your Data Stays Private</span></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="card"><div class="card-icon">📄</div><b>PDF Report</b><span>Download Instant Report</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_cancer_cnn_model.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def create_pdf_report(image_pil, result, confidence, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(4, 47, 46)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",22)
    pdf.set_xy(10,10); pdf.cell(190,10,"SkinSense AI",align="C")
    pdf.set_font("Helvetica","",10)
    pdf.set_xy(10,23); pdf.cell(190,8,"Early Detection | Benign vs Malignant | Powered by Deep Learning",align="C")
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(253,230,138)
    pdf.set_xy(10,33); pdf.cell(190,8,"Developed by Abdul Rehman - AR",align="C")
    img_bytes = io.BytesIO(); image_pil.save(img_bytes, format="JPEG"); img_bytes.seek(0)
    pdf.image(img_bytes, x=65, y=55, w=80, h=80)
    y=145
    if result=="Malignant":
        pdf.set_fill_color(254,242,242); pdf.set_draw_color(248,113,113); pdf.set_text_color(185,28,28)
        label="MALIGNANT - Consult Doctor Immediately"
    else:
        pdf.set_fill_color(236,253,245); pdf.set_draw_color(110,231,183); pdf.set_text_color(4,120,87)
        label="BENIGN - No Cancer Detected"
    pdf.set_xy(15,y); pdf.set_font("Helvetica","B",15); pdf.cell(180,15,label,border=1,fill=True,align="C")
    pdf.set_text_color(15,23,42)
    details=[("Date & Time:",datetime.now().strftime('%d-%m-%Y %H:%M')),("File Name:",filename),("Prediction:",result),("Confidence:",f"{confidence:.2f}%"),("Model:","CNN (128x128)"),("Developed by:","Abdul Rehman")]
    y+=28
    for k,v in details:
        pdf.set_xy(20,y); pdf.set_font("Helvetica","B",12); pdf.cell(45,8,k)
        pdf.set_font("Helvetica","",12); pdf.cell(120,8,v); y+=10
    y+=8
    pdf.set_xy(15,y); pdf.set_fill_color(255,251,235); pdf.set_draw_color(253,230,138); pdf.set_text_color(146,64,14)
    pdf.set_font("Helvetica","",10)
    pdf.multi_cell(180,7,"Disclaimer: This report is for educational purposes only. It is not a final medical diagnosis. Please consult a Dermatologist.",border=1,fill=True,align="C")
    return bytes(pdf.output())

st.markdown("### Upload Skin Lesion Image")
st.caption("Upload a clear, close-up image of the skin lesion for best accuracy.")
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Your Uploaded Image', use_container_width=True)
    resized_image = image.resize((128,128))
    img_array = np.array(resized_image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('🔍  Analyze Now'):
        with st.spinner('AI Analyzing...'):
            prediction = model.predict(img_array)[0][0]
        if prediction > 0.5:
            result="Malignant"; confidence=prediction*100
            st.markdown(f'<div class="result-malignant"><b style="color:#B91C1C; font-size:18px;">⚠️ Prediction: Malignant</b><br><span>Confidence: {confidence:.2f}% • Risk: High</span><br><span>Please consult a doctor immediately.</span></div>', unsafe_allow_html=True)
        else:
            result="Benign"; confidence=(1-prediction)*100
            st.markdown(f'<div class="result-benign"><b style="color:#047857; font-size:18px;">✅ Prediction: Benign</b><br><span>Confidence: {confidence:.2f}% • Risk: Low</span><br><span>Good news! It looks Benign. Please still consult a doctor for confirmation.</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(confidence))
        st.session_state['pdf_report'] = create_pdf_report(image, result, confidence, uploaded_file.name)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('🗑️ Clear Image'):
            st.session_state.clear(); st.rerun()
    with col2:
        if 'pdf_report' in st.session_state:
            st.download_button(label="📥  Download PDF Report", data=st.session_state['pdf_report'], file_name="SkinSense_AI_Report.pdf", mime="application/pdf")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ Disclaimer: This tool is for educational purposes only. It is not a final medical diagnosis.</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#9CA3AF; font-size:12px; margin-top:12px;'>SkinSense AI • Developed by <b>Abdul Rehman</b> • Final Year Project</p>", unsafe_allow_html=True)
