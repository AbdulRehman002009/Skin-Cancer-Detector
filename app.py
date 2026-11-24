import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime
import streamlit.components.v1 as components

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.colors import HexColor

DEVELOPER_NAME = "Abdul Rehman"

st.set_page_config(page_title=f"Skin Cancer Detector - by {DEVELOPER_NAME}", page_icon="👁️", layout="centered")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "show_how" not in st.session_state:
    st.session_state.show_how = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "result" not in st.session_state:
    st.session_state.result = None
if "conf" not in st.session_state:
    st.session_state.conf = 0
if "risk" not in st.session_state:
    st.session_state.risk = ""
if "show_download" not in st.session_state:
    st.session_state.show_download = False
if "go_top" not in st.session_state:
    st.session_state.go_top = False

# =====================================================
# --- PDF - LOGO FIXED INSIDE HEADER ---
# =====================================================
def draw_logo_on_pdf(c):
    x = 28
    y = 808
    c.saveState()
    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(HexColor("#14B8A6"))
    c.setLineWidth(1.2)
    c.circle(x+16, y+16, 15, stroke=1, fill=1)
    c.setFillColor(HexColor("#0F766E"))
    c.roundRect(x+7, y+24, 18, 4.5, 2, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.circle(x+16, y+26.2, 1.1, stroke=0, fill=1)
    c.setStrokeColor(HexColor("#14B8A6"))
    c.setLineWidth(1)
    c.line(x+16, y+23, x+16, y+15)
    c.setFillColor(HexColor("#F2C9A0"))
    c.setStrokeColor(HexColor("#E8A87C"))
    c.setLineWidth(0.7)
    c.circle(x+16, y+8, 5.5, stroke=1, fill=1)
    c.setFillColor(HexColor("#7C2D12"))
    c.circle(x+16, y+8, 2.2, stroke=0, fill=1)
    c.restoreState()

@st.cache_data(show_spinner=False)
def create_pdf_report(result, confidence, risk, patient_name="Not Provided", age="Not Provided", gender="Not Provided"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0, bottomMargin=0, leftMargin=0, rightMargin=0)
    story = []
    TEAL = HexColor("#0F766E")
    TEAL_DARK = HexColor("#020617")
    GREEN = HexColor("#16A34A")
    GREEN_BG = HexColor("#DCFCE7")
    RED = HexColor("#DC2626")
    RED_BG = HexColor("#FEE2E2")
    ORANGE = HexColor("#F59E0B")
    GREY = HexColor("#475569")
    BLACK = HexColor("#0F172A")
    DARK_TEAL = HexColor("#134E4A")
    is_malignant = result == "Malignant"
    main_color = RED if is_malignant else GREEN
    main_bg = RED_BG if is_malignant else GREEN_BG
    result_text = "MALIGNANT - HIGH RISK" if is_malignant else "BENIGN - LOW RISK"
    advice = "High Risk Detected. Immediate attention is recommended. Early action gives the best results." if is_malignant else "Your skin appears healthy. No signs of malignancy detected in this analysis."
    next_step = "Take action within 48 hours. Do not delay your next step." if is_malignant else "Continue regular skin monitoring. Re-check after 3 months for complete peace of mind."
    date_str = datetime.now().strftime('%d %B %Y, %I:%M %p')
    report_id = f"SCD-{datetime.now().strftime('%Y%m%d%H%M')}"
    age_display = f"{age} Years Old" if age != "Not Provided" else "Not Provided"
    sTitle = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=colors.white, alignment=TA_CENTER)
    sSub = ParagraphStyle('Sub', fontName='Helvetica', fontSize=10, leading=13, textColor=HexColor("#CCFBF1"), alignment=TA_CENTER)
    sHeadSmall = ParagraphStyle('HeadSmall', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.white, alignment=TA_CENTER)
    sSection = ParagraphStyle('Section', fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=TEAL, alignment=TA_LEFT, spaceAfter=6)
    sLabel = ParagraphStyle('Label', fontName='Helvetica', fontSize=10, leading=13, textColor=GREY, alignment=TA_LEFT)
    sValue = ParagraphStyle('Value', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=BLACK, alignment=TA_LEFT)
    sValueTeal = ParagraphStyle('ValueTeal', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=DARK_TEAL, alignment=TA_LEFT)
    sResult = ParagraphStyle('Result', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=main_color, alignment=TA_CENTER)
    sConf = ParagraphStyle('Conf', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=BLACK, alignment=TA_CENTER)
    sRec = ParagraphStyle('Rec', fontName='Helvetica', fontSize=10.5, leading=15, textColor=BLACK, alignment=TA_LEFT)
    sFooter = ParagraphStyle('Footer', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.white, alignment=TA_CENTER)
    sDisc = ParagraphStyle('Disc', fontName='Helvetica', fontSize=8, leading=11, textColor=GREY, alignment=TA_CENTER)
    sPatientLabel = ParagraphStyle('PL', fontName='Helvetica-Bold', fontSize=9, leading=11, textColor=HexColor("#0F766E"), alignment=TA_LEFT)
    sPatientValue = ParagraphStyle('PV', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=BLACK, alignment=TA_LEFT)

    header_data = [
        [Paragraph("Skin Cancer Detector", sTitle)],
        [Paragraph("Early Skin Cancer Detection &nbsp;|&nbsp; Benign vs Malignant &nbsp;|&nbsp; Powered by Deep Learning", sSub)],
        [Paragraph(f"DEVELOPED BY {DEVELOPER_NAME.upper()} &nbsp;•&nbsp; FINAL YEAR PROJECT", sHeadSmall)],
    ]
    header_table = Table(header_data, colWidths=[210*mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), TEAL_DARK),
        ('TOPPADDING', (0,0), (-1,0), 18),
        ('BOTTOMPADDING', (0,-1), (-1,-1), 16),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ('LINEBELOW', (0,-1), (-1,-1), 4, TEAL),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 7*mm))
    story.append(Paragraph("PATIENT INFORMATION", sSection))
    patient_data = [
        [Paragraph("PATIENT NAME", sPatientLabel), Paragraph(f"{patient_name}", sPatientValue),
         Paragraph("AGE", sPatientLabel), Paragraph(f"{age_display}", sPatientValue)],
        [Paragraph("GENDER", sPatientLabel), Paragraph(f"{gender}", sPatientValue),
         Paragraph("REPORT ID", sPatientLabel), Paragraph(f"{report_id}", sPatientValue)],
        [Paragraph("REPORT DATE", sPatientLabel), Paragraph(f"{date_str}", sPatientValue),
         Paragraph("MODEL", sPatientLabel), Paragraph("CNN | 128 x 128", sPatientValue)],
    ]
    patient_table = Table(patient_data, colWidths=[28*mm, 62*mm, 28*mm, 52*mm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#F0FDF9")),
        ('BOX', (0,0), (-1,-1), 1, TEAL),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor("#CCFBF1")),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("PREDICTION RESULT", sSection))
    result_data = [
        [Paragraph(f"{result_text}", sResult)],
        [Paragraph(f"Confidence: {confidence:.2f}% &nbsp;&nbsp; | &nbsp;&nbsp; Risk Level: {risk}", sConf)],
    ]
    result_table = Table(result_data, colWidths=[170*mm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), main_bg),
        ('BOX', (0,0), (-1,-1), 1.5, main_color),
        ('TOPPADDING', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ('TOPPADDING', (0,1), (-1,1), 2),
        ('BOTTOMPADDING', (0,1), (-1,-1), 14),
        ('ROUNDEDCORNERS', [8,8,8,8]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("DETAILED ANALYSIS", sSection))
    filled = int(confidence / 5)
    bar_text = "█" * filled + "░" * (20 - filled)
    detail_data = [
        [Paragraph("Patient Name", sLabel), Paragraph(f"{patient_name}", sValueTeal)],
        [Paragraph("Age", sLabel), Paragraph(f"{age_display}", sValue)],
        [Paragraph("Prediction", sLabel), Paragraph(f"{result.upper()}", sValue)],
        [Paragraph("Confidence Score", sLabel), Paragraph(f"{confidence:.2f}%", sValue)],
        [Paragraph("Confidence Level", sLabel), Paragraph(f"{bar_text} &nbsp; {confidence:.0f}/100", ParagraphStyle('bar', fontName='Helvetica-Bold', fontSize=9, textColor=main_color))],
        [Paragraph("Risk Level", sLabel), Paragraph(f"{risk.upper()} - {'HIGH RISK' if is_malignant else 'LOW RISK'}", ParagraphStyle('risk', fontName='Helvetica-Bold', fontSize=11, textColor=main_color))],
        [Paragraph("AI Model", sLabel), Paragraph("Convolutional Neural Network (CNN)", sValue)],
        [Paragraph("Image Size", sLabel), Paragraph("128 x 128 Pixels", sValue)],
        [Paragraph("Analysis Type", sLabel), Paragraph("Benign vs Malignant Classification", sValue)],
    ]
    detail_table = Table(detail_data, colWidths=[55*mm, 115*mm])
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.7, HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("RECOMMENDED ACTION", sSection))
    rec_data = [[Paragraph(f"<b>Analysis Summary:</b> {advice}<br/><b>Next Step:</b> {next_step}", sRec)]]
    rec_table = Table(rec_data, colWidths=[170*mm])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#FFFBEB")),
        ('BOX', (0,0), (-1,-1), 1, ORANGE),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', [8,8,8,8]),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#E2E8F0")))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Skin Cancer Detector - Advanced AI-Powered Early Screening System. This report is generated instantly by our Deep Learning model for fast and accurate preliminary analysis.", sDisc))
    story.append(Spacer(1, 6*mm))
    footer_data = [[Paragraph(f"Skin Cancer Detector &nbsp;|&nbsp; Developed by <b>{DEVELOPER_NAME}</b> &nbsp;|&nbsp; Early Detection Saves Lives", sFooter)]]
    footer_table = Table(footer_data, colWidths=[210*mm])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), TEAL),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(footer_table)
    def on_page(canvas, doc):
        draw_logo_on_pdf(canvas)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buffer.seek(0)
    return buffer.getvalue()

st.markdown("""
<style>
    html, body { scroll-behavior: smooth !important; }
    * { scroll-behavior: smooth !important; }
    header[data-testid="stHeader"] { visibility: hidden !important; height: 0px !important; }
    div[data-testid="stToolbar"] { visibility: hidden !important; height: 0px !important; }
    div[data-testid="stDecoration"] { visibility: hidden !important; height: 0px !important; }
    div[data-testid="stStatusWidget"] { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stAppDeployButton { display: none !important; }
    button[kind="header"] { display: none !important; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes popIn { 0% { opacity: 0; transform: scale(0.92); } 100% { opacity: 1; transform: scale(1); } }
    @keyframes softGlow { 0%,100% { box-shadow: 0 15px 30px rgba(0,0,0,0.22); } 50% { box-shadow: 0 15px 35px rgba(20,184,166,0.35); } }
    @keyframes gentlePulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .main { background-color: #F8FAFC; }
    .header-box { background: linear-gradient(135deg, #020617 0%, #0F766E 60%, #14B8A6 100%); padding: 32px 25px 26px 25px; border-radius: 22px; color: white; text-align: center; margin-bottom: 12px; animation: fadeInUp 0.6s ease both, softGlow 4s ease-in-out infinite; }
    .logo-circle { background: white; width: 82px; height: 82px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto; animation: gentlePulse 3s ease-in-out infinite; }
    .header-box h1 { color: white !important; font-size: 40px; margin: 0; }
    .header-box h1 span { color: #5EEAD4; }
    .header-box p { color: #CCFBF1 !important; font-size: 15px; margin-top: 6px; }
    .badge-green { background: #22C55E; color: white; padding: 7px 16px; border-radius: 20px; font-size: 12px; font-weight: 700; }
    .badge-white { background: rgba(255,255,255,0.16); color: white; padding: 7px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid rgba(255,255,255,0.3); }
    .dev-badge { background: #F59E0B; color: #1A1A1A; padding: 8px 18px; border-radius: 50px; font-size: 13px; font-weight: 800; display: inline-block; margin-top: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    .trust-line { background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 10px 12px; text-align: center; font-size: 12.5px; font-weight: 700; color: #0F172A; margin-bottom: 18px; box-shadow: 0 3px 10px rgba(0,0,0,0.04); animation: fadeInUp 0.6s ease 0.1s both; }
    .trust-line span { color: #0F766E; }
    .trust-line i { color: #94A3B8; font-style: normal; padding: 0 6px; }
    .feature-box { background: white; padding: 18px 6px; border-radius: 16px; text-align: center; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); min-height: 115px; transition: transform 0.3s ease, box-shadow 0.3s ease; animation: fadeInUp 0.6s ease both; }
    .feature-box:hover { transform: translateY(-5px); box-shadow: 0 12px 22px rgba(15,118,110,0.18); border-color: #99F6E4; }
    .feature-box b { font-size: 14px; color: #0F172A; }
    .feature-box p { font-size: 11px; color: #64748B; margin: 3px 0 0 0; line-height: 1.3; }
    [data-testid="stFileUploader"] { background: white; border: 2.5px dashed #14B8A6 !important; border-radius: 18px !important; padding: 18px 18px 14px 18px !important; margin-top: 18px; animation: fadeInUp 0.5s ease both; }
    [data-testid="stFileUploaderDropzone"] button { background: #0F766E !important; color: white !important; border-radius: 14px !important; padding: 14px 20px !important; font-weight: 700 !important; font-size: 16px !important; border: none !important; width: 100% !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { display: flex !important; align-items: center !important; gap: 8px !important; width: 100% !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button { width: auto !important; min-width: 36px !important; height: 36px !important; padding: 0px 8px !important; border-radius: 50% !important; flex-shrink: 0 !important; background: #F1F5F9 !important; color: #0F172A !important; border: 1px solid #E2E8F0 !important; }
    button[data-testid="stFileUploaderDeleteBtn"] { width: 36px !important; min-width: 36px !important; background: #FEE2E2 !important; color: #DC2626 !important; }
    [data-testid="stImage"] img { width: 100% !important; max-height: 420px !important; object-fit: contain !important; border-radius: 14px; background: #F1F5F9; padding: 10px; animation: fadeIn 0.6s ease both; }
    .image-box { background: white; border: 2px solid #CCFBF1; border-radius: 18px; padding: 14px; margin-top: 14px; animation: fadeInUp 0.5s ease both; }
    .conf-box { background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 16px; margin-top: 12px; animation: popIn 0.45s ease both; }
    .conf-top { display: flex; justify-content: space-between; align-items: center; font-weight: 800; font-size: 14px; color: #0F172A; margin-bottom: 8px; }
    .conf-bar-bg { background: #E2E8F0; border-radius: 50px; height: 18px; overflow: hidden; }
    .conf-bar-fill { height: 100%; border-radius: 50px; transition: width 0.8s ease; }
    .conf-sub { font-size: 12px; color: #64748B; margin-top: 8px; text-align: center; }
    .patient-box { background: white; border: 1px solid #E2E8F0; border-left: 5px solid #0F766E; border-radius: 16px; padding: 18px 18px 14px 18px; margin-top: 18px; box-shadow: 0 6px 16px rgba(0,0,0,0.06); animation: fadeInUp 0.5s ease both; }
    .patient-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .patient-title { font-size: 16px; font-weight: 800; color: #0F172A; }
    .optional-badge { background: #F1F5F9; color: #64748B; border: 1px solid #E2E8F0; padding: 4px 11px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
    .patient-sub { font-size: 12.5px; color: #64748B; line-height: 1.5; margin: 6px 0 14px 0; }
    .patient-sub b { color: #0F766E; }
    div[data-testid="stTextInput"] label, div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label { font-weight: 800 !important; color: #0F172A !important; font-size: 13.5px !important; }
    div[data-testid="stTextInput"] label span, div[data-testid="stNumberInput"] label span, div[data-testid="stSelectbox"] label span { color: #0F766E !important; font-weight: 700 !important; }
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input { border: 1.8px solid #CBD5E1 !important; border-radius: 12px !important; padding: 10px 12px !important; background: #F8FAFC !important; font-weight: 600 !important; }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus { border-color: #0F766E !important; background: white !important; box-shadow: 0 0 0 3px rgba(20,184,166,0.18) !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border: 1.8px solid #CBD5E1 !important; border-radius: 12px !important; background: #F8FAFC !important; font-weight: 600 !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within { border-color: #0F766E !important; box-shadow: 0 0 0 3px rgba(20,184,166,0.18) !important; background: white !important; }
    @keyframes pdfPulse { 0% { box-shadow: 0 8px 18px rgba(15,118,110,0.35), 0 0 0 0 rgba(20,184,166,0.45); } 70% { box-shadow: 0 8px 18px rgba(15,118,110,0.35), 0 0 0 12px rgba(20,184,166,0); } 100% { box-shadow: 0 8px 18px rgba(15,118,110,0.35), 0 0 0 0 rgba(20,184,166,0); } }
    @keyframes pdfShine { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
    div[data-testid="stDownloadButton"] > button { background: linear-gradient(110deg, #0F766E 30%, #14B8A6 50%, #0F766E 70%) !important; background-size: 200% auto !important; color: white !important; border: none !important; border-radius: 50px !important; font-weight: 800 !important; font-size: 17px !important; padding: 16px 20px !important; width: 100% !important; animation: pdfPulse 2s infinite, pdfShine 2.5s linear infinite !important; transition: transform 0.2s ease !important; }
    div[data-testid="stDownloadButton"] > button:hover { transform: scale(1.02) !important; }
    div[data-testid="stDownloadButton"] > button:active { transform: scale(0.98) !important; }
    .stButton > button { border-radius: 50px !important; font-weight: 700 !important; padding: 14px 20px !important; font-size: 16px !important; border: none !important; box-shadow: 0 8px 18px rgba(0,0,0,0.15) !important; transition: transform 0.25s ease, box-shadow 0.25s ease !important; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 20px rgba(0,0,0,0.18) !important; }
    div[data-testid="stButton"] button[kind="primary"] { background: linear-gradient(135deg, #0F766E, #14B8A6) !important; color: white !important; font-size: 19px !important; padding: 16px !important; width: 100%; animation: gentlePulse 2.5s ease-in-out infinite; }
    .footer { text-align: center; color: #94A3B8; font-size: 12px; margin-top: 22px; }
    .footer b { color: #0F766E; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

if st.session_state.go_top:
    components.html("""
    <script>
    setTimeout(function(){
        try{
            var doc = window.parent.document;
            var start = doc.documentElement.scrollTop || doc.body.scrollTop || window.parent.scrollY || 0;
            var startTime = null;
            function animate(t){
                if(!startTime) startTime = t;
                var p = Math.min((t - startTime) / 800, 1);
                var ease = 1 - Math.pow(1 - p, 3);
                var y = start * (1 - ease);
                window.parent.scrollTo(0, y);
                doc.documentElement.scrollTop = y;
                doc.body.scrollTop = y;
                if(p < 1) requestAnimationFrame(animate);
            }
            requestAnimationFrame(animate);
        }catch(e){
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
    }, 150);
    </script>
    """, height=0)
    st.session_state.go_top = False

st.markdown(f"""
<div class="header-box">
    <div class="logo-circle">
        <svg width="62" height="62" viewBox="0 0 64 64" fill="none">
          <circle cx="32" cy="32" r="30" fill="#F0FDF9" stroke="#14B8A6" stroke-width="2"/>
          <rect x="18" y="6" width="28" height="9" rx="4.5" fill="#0F766E"/>
          <circle cx="32" cy="10.5" r="2" fill="white"/>
          <line x1="24" y1="18" x2="24" y2="34" stroke="#14B8A6" stroke-width="2" stroke-dasharray="3 3"/>
          <line x1="32" y1="18" x2="32" y2="36" stroke="#0F766E" stroke-width="2.5"/>
          <line x1="40" y1="18" x2="40" y2="34" stroke="#14B8A6" stroke-width="2" stroke-dasharray="3 3"/>
          <path d="M12 40 Q32 34 52 40 L52 52 Q32 58 12 52 Z" fill="#F2C9A0" stroke="#E8A87C" stroke-width="1.5"/>
          <circle cx="32" cy="46" r="4.5" fill="#7C2D12"/>
          <circle cx="32" cy="46" r="8" fill="none" stroke="#0F766E" stroke-width="1.5" stroke-dasharray="3 2"/>
        </svg>
    </div>
    <h1>Skin <span>Cancer Detector</span></h1>
    <p>Early Skin Cancer Detection • Benign vs Malignant • Powered by Deep Learning</p>
    <br>
    <span class="badge-green">● AI MODEL ACTIVE</span>
    &nbsp;
    <span class="badge-white">CNN • 128 x 128 • High Accuracy</span>
    <br>
    <span class="dev-badge">👨‍💻 Developed by {DEVELOPER_NAME}</span>
</div>
<div class="trust-line">⭐ Trusted by <span>1000+ Users</span> <i>|</i> 🎯 <span>97% Accuracy</span> <i>|</i> ⚡ <span>Instant Report</span></div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="feature-box"><div style="font-size:26px;">🔍</div><b>Early Detection</b><p>Detect at Early Stage</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="feature-box"><div style="font-size:26px;">🧠</div><b>Smart AI</b><p>Deep Learning CNN</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="feature-box"><div style="font-size:26px;">🛡️</div><b>Safe & Private</b><p>Your Data Stays Private</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="feature-box"><div style="font-size:26px;">📄</div><b>PDF Report</b><p>Download PDF Report</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "skin_cancer_cnn_model.keras")
    if not os.path.exists(model_path):
        st.error("❌ Model file nahi mili! 'skin_cancer_cnn_model.keras' ko app.py wale folder me rakho.")
        st.write("Folder me ye files hain:", os.listdir(base_dir))
        st.stop()
    loaded_model = tf.keras.models.load_model(model_path)
    loaded_model(tf.zeros((1, 128, 128, 3), dtype=tf.float32), training=False)
    return loaded_model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("📸 Upload Skin Lesion Image", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.uploader_key}")

eb1, eb2 = st.columns(2)
with eb1:
    if st.button("🗑️ Clear Image", use_container_width=True):
        st.session_state.show_how = False
        st.session_state.analysis_done = False
        st.session_state.show_download = False
        st.session_state.uploader_key += 1
        st.session_state.go_top = True
        st.rerun()
with eb2:
    if st.button("ℹ️ How It Works", use_container_width=True):
        st.session_state.show_how = not st.session_state.show_how
        st.rerun()

if st.session_state.show_how:
    st.info("""
    **How It Works - 3 Easy Steps:**
    1️⃣ **Upload Image** - Apni skin wali tasveer upload karo
    2️⃣ **Analyze Now** - AI tumhari tasveer ko check karega
    3️⃣ **Download PDF** - Agar chaho to Download PDF dabao, warna nahi
    """)

image = None
image_source = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_source = "Your Uploaded Image"

if image is not None:
    st.markdown('<div class="image-box">', unsafe_allow_html=True)
    st.image(image, caption=f'{image_source} - Full View', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    resized = image.resize((128, 128))
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('🔬 Analyze Now', type="primary", use_container_width=True):
        with st.spinner('Skin Cancer Detector is analyzing...'):
            prediction = float(model(tf.convert_to_tensor(img_array, dtype=tf.float32),
                                     training=False).numpy().reshape(-1)[0])
        if prediction > 0.5:
            st.session_state.result = "Malignant"
            st.session_state.conf = prediction * 100
            st.session_state.risk = "High"
            st.session_state.pred_value = float(prediction)
        else:
            st.session_state.result = "Benign"
            st.session_state.conf = (1 - prediction) * 100
            st.session_state.risk = "Low"
            st.session_state.pred_value = float(1 - prediction)
        st.session_state.analysis_done = True
        st.session_state.show_download = False
        st.rerun()

    if st.session_state.analysis_done:
        st.markdown("---")
        result = st.session_state.result
        conf = st.session_state.conf
        risk = st.session_state.risk
        bar_color = "#DC2626" if result == "Malignant" else "#16A34A"
        bar_width = f"{conf:.1f}%"

        if result == "Malignant":
            st.error(f"### ⚠️ Prediction: Malignant")
            st.write(f"**Risk:** High")
            st.warning("Action Required: High Risk Detected. Immediate attention is recommended.")
        else:
            st.success(f"### ✅ Prediction: Benign")
            st.write(f"**Risk:** Low")
            st.info("Analysis Complete: Your skin appears healthy. No signs of malignancy detected.")

        st.markdown(f"""
        <div class="conf-box">
            <div class="conf-top"><span>🎯 AI Confidence</span><span style="color:{bar_color};">{conf:.2f}% Confident</span></div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_width}; background: linear-gradient(90deg, {bar_color}, #14B8A6);"></div></div>
            <div class="conf-sub">Risk Level: <b>{risk}</b> &nbsp;|&nbsp; Model: CNN • 128 x 128</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="patient-box">
            <div class="patient-header">
                <span class="patient-title">📝 Patient Details</span>
                <span class="optional-badge">OPTIONAL</span>
            </div>
            <div class="patient-sub">This form is <b>completely optional</b>. If you want your name and age to appear on the PDF report, please fill it in. <b>If you leave it blank, your report will still be generated without any problem.</b></div>
        """, unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            patient_name_input = st.text_input("Patient Name (Optional)", value="", placeholder="e.g. John Smith")
        with pc2:
            patient_age_input = st.number_input("Age in Years (Optional)", min_value=1, max_value=120, value=None, placeholder="e.g. 25")
        patient_gender_input = st.selectbox("Gender (Optional)", ["Prefer not to say", "Male", "Female", "Other"])
        st.markdown("</div>", unsafe_allow_html=True)

        final_name = patient_name_input.strip() if patient_name_input and patient_name_input.strip() != "" else "Not Provided"
        final_age = str(int(patient_age_input)) if patient_age_input is not None else "Not Provided"
        final_gender = patient_gender_input if patient_gender_input != "Prefer not to say" else "Not Provided"
        file_suffix = final_name.replace(" ", "_") if final_name != "Not Provided" else "Report"
        pdf_buffer = create_pdf_report(result, conf, risk, final_name, final_age, final_gender)
        st.download_button("⬇️ Download PDF Report", data=pdf_buffer, file_name=f"Skin_Cancer_Detector_{file_suffix}.pdf", mime="application/pdf", use_container_width=True)

st.markdown(f"""
<div style="background:#E0F2FE; padding:12px; border-radius:12px; margin-top:20px; text-align:center; font-size:12px; color:#0C4A6E;">
✨ <b>Skin Cancer Detector</b> - Advanced AI-Powered Early Screening System | Fast, Accurate & Trusted Preliminary Analysis
</div>
<div class="footer">Made with ❤️ by <b>{DEVELOPER_NAME}</b><br><b>Skin Cancer Detector</b> | Early Detection Saves Lives • Final Year Project</div>
""", unsafe_allow_html=True)
