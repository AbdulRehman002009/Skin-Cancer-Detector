import streamlit as st
import numpy as np
from PIL import Image
import io
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib import colors

# ===== COLORS - Pehle wale hi =====
TEAL = HexColor("#0F766E")
LIGHT_TEAL = HexColor("#14B8A6")
LIGHT_BG = HexColor("#F0FDF9")
SKIN = HexColor("#F2C9A0")
SKIN_DARK = HexColor("#E8A87C")
DARK_TEXT = HexColor("#134E4A")
GREY = HexColor("#64748B")
LIGHT_GREY = HexColor("#E2E8F0")
RED = HexColor("#DC2626")
GREEN = HexColor("#059669")
ORANGE = HexColor("#EA580C")

DEVELOPER_NAME = "Ahmad Raza"
DEVELOPER_EMAIL = "ahmad.raza@example.com"

st.set_page_config(page_title="SkinSense AI", page_icon="🧬", layout="wide")

# ===== 1. SIRF LOGO CHANGE - WEB HEADER =====
st.markdown("""
<div style="display:flex; align-items:center; gap:12px;">
  <div style="width:58px; height:58px;">
    <svg width="58" height="58" viewBox="0 0 64 64" fill="none">
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
  <div>
    <div style="font-size:32px; font-weight:800; color:#0F766E; line-height:1;">SkinSense <span style="color:#14B8A6;">AI</span></div>
    <div style="font-size:13px; color:#64748B; margin-top:4px; letter-spacing:1.5px; font-weight:600;">SKIN CANCER DETECTION • AI POWERED</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #CCFBF1; margin:16px 0;'>", unsafe_allow_html=True)

# ===== 2. SIRF LOGO CHANGE - PDF =====
def draw_logo_on_pdf(c, x, y):
    c.saveState()
    c.setFillColor(HexColor("#F0FDF9"))
    c.setStrokeColor(LIGHT_TEAL)
    c.setLineWidth(1.5)
    c.circle(x+18, y+18, 17, stroke=1, fill=1)
    c.setFillColor(TEAL)
    c.roundRect(x+8, y+27, 20, 5, 2, stroke=0, fill=1)
    c.setFillColor(white)
    c.circle(x+18, y+29.5, 1.2, stroke=0, fill=1)
    c.setStrokeColor(LIGHT_TEAL)
    c.setLineWidth(1.2)
    c.line(x+18, y+26, x+18, y+17)
    c.setFillColor(SKIN)
    c.circle(x+18, y+9, 6, stroke=0, fill=1)
    c.setFillColor(HexColor("#7C2D12"))
    c.circle(x+18, y+9, 2.5, stroke=0, fill=1)
    c.restoreState()

# ===== PDF - Pehle wala hi, sirf logo add hua =====
def create_pdf_report(patient_name, age, gender, prediction, confidence, image_bytes, notes=""):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle('Title2', parent=styles['Title'], fontSize=22, textColor=TEAL, alignment=TA_LEFT, spaceAfter=2, fontName='Helvetica-Bold')
    s_sub = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9, textColor=GREY, alignment=TA_LEFT, fontName='Helvetica', spaceAfter=6)
    s_h = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12, textColor=TEAL, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6)
    s_body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, textColor=DARK_TEXT, fontName='Helvetica', leading=14)
    s_small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8.5, textColor=GREY, fontName='Helvetica', leading=11)

    story = []
    header_data = [
        [Paragraph("<b><font color='#0F766E' size=20>SkinSense AI</font></b><br/><font color='#64748B' size=8>SKIN CANCER DETECTION • AI POWERED</font>", s_body),
         Paragraph(f"<font color='#64748B' size=8>Report ID: SS-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}<br/>Date: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}<br/>Developed by: {DEVELOPER_NAME}</font>", s_body)]
    ]
    t = Table(header_data, colWidths=[90*mm, 75*mm])
    t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t)
    story.append(HRFlowable(width="100%", thickness=1.5, color=LIGHT_TEAL, spaceAfter=8, spaceBefore=6))
    story.append(Paragraph("Skin Cancer Detection Report", s_title))
    story.append(Paragraph("This report is generated by AI for screening purposes only. It is not a final medical diagnosis. Please consult a certified dermatologist.", s_sub))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Patient Information", s_h))
    info = [
        [Paragraph("<b>Patient Name</b>", s_body), Paragraph(patient_name, s_body), Paragraph("<b>Age</b>", s_body), Paragraph(str(age), s_body)],
        [Paragraph("<b>Gender</b>", s_body), Paragraph(gender, s_body), Paragraph("<b>Report Date</b>", s_body), Paragraph(datetime.datetime.now().strftime("%d-%m-%Y"), s_body)],
    ]
    ti = Table(info, colWidths=[32*mm, 50*mm, 32*mm, 51*mm])
    ti.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), LIGHT_BG), ('BOX', (0,0), (-1,-1), 0.7, LIGHT_TEAL), ('INNERGRID', (0,0), (-1,-1), 0.4, LIGHT_GREY), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6), ('LEFTPADDING', (0,0), (-1,-1), 8)]))
    story.append(ti)

    story.append(Paragraph("Analysis Result", s_h))
    result_color = GREEN if "Benign" in prediction else RED if "Malignant" in prediction else ORANGE
    img_elem = None
    if image_bytes:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_img.thumbnail((400, 400))
            tmp = io.BytesIO()
            pil_img.save(tmp, format="JPEG", quality=90)
            tmp.seek(0)
            img_elem = RLImage(tmp, width=55*mm, height=55*mm)
            img_elem.hAlign = 'CENTER'
        except:
            img_elem = Paragraph("Image not available", s_small)

    right_content = [
        Paragraph("Predicted Condition", ParagraphStyle('l', parent=s_body, fontSize=9, textColor=GREY)),
        Paragraph(f"<b><font color='{result_color.hexval()}'>{prediction}</font></b>", ParagraphStyle('r', parent=s_body, fontSize=16, leading=20)),
        Spacer(1, 3*mm),
        Paragraph(f"<b>Confidence: {confidence:.1f}%</b>", s_body),
        Spacer(1, 3*mm),
        Paragraph("Risk Level: " + ("LOW - Routine Checkup" if "Benign" in prediction else "HIGH - Immediate Dermatologist Visit" if "Malignant" in prediction else "MODERATE - Follow-up Required"), s_body),
    ]
    if img_elem:
        combo = Table([[img_elem, right_content]], colWidths=[65*mm, 100*mm])
        combo.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOX', (0,0), (-1,-1), 0.7, LIGHT_TEAL), ('BACKGROUND', (0,0), (-1,-1), white), ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8), ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8)]))
        story.append(combo)

    story.append(Paragraph("Recommendations", s_h))
    if "Malignant" in prediction:
        recs = ["1. Foran kisi certified dermatologist se rabta karein, is report ko sath le jayein.", "2. Is jagah ko khud se na cherein, na dabayein aur na koi cream lagayein.", "3. Dhoop se bachain aur 2 haftay ke andar dobara checkup karwayein."]
    elif "Benign" in prediction:
        recs = ["1. Ghabrane ki zaroorat nahi, lekin har 3 mahine baad dobara check karwayein.", "2. Agar til ka size, rang ya shape badle to dobara scan karwayein.", "3. Dhoop me SPF 50+ sunscreen istemal karein."]
    else:
        recs = ["1. Dermatologist se follow-up checkup zaroor karwayein.", "2. Is jagah ki clear photo le kar record rakhein."]
    for r in recs:
        story.append(Paragraph(r, ParagraphStyle('rec', parent=s_body, leftIndent=6, spaceAfter=3)))
    if notes:
        story.append(Paragraph("Doctor / User Notes", s_h))
        story.append(Paragraph(notes, s_body))
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=LIGHT_GREY, spaceAfter=4, spaceBefore=4))
    story.append(Paragraph(f"Generated by SkinSense AI | Developer: {DEVELOPER_NAME} | Contact: {DEVELOPER_EMAIL} | This is an AI screening tool, not a replacement for professional medical advice.", s_small))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("___________________________ &nbsp;&nbsp;&nbsp;&nbsp; ___________________________<br/>Patient Signature &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dermatologist Signature", s_small))

    def on_page(canvas, doc):
        canvas.saveState()
        draw_logo_on_pdf(canvas, 480, 795)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawCentredString(297, 28, f"SkinSense AI - Confidential Medical Report | Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buffer.seek(0)
    return buffer.getvalue()

# ===== MAIN APP - Pehle wala hi layout =====
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Patient Details")
    patient_name = st.text_input("Patient Name", value="Ahmad Raza")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    notes = st.text_area("Notes (Optional)", placeholder="Koi extra note...")
with col2:
    st.subheader("Upload Skin Image")
    uploaded = st.file_uploader("Skin ki photo upload karein", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)

if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
    if not uploaded:
        st.warning("Pehle photo upload karein!")
    else:
        # Yahan tumhara original model prediction aayega
        prediction = "Benign (Non-Cancerous)"
        confidence = 92.5
        st.success(f"Result: {prediction} - Confidence: {confidence}%")
        st.progress(confidence/100)
        image_bytes = uploaded.getvalue()
        pdf_bytes = create_pdf_report(patient_name, age, gender, prediction, confidence, image_bytes, notes)
        st.download_button("📄 Download Improved PDF Report", data=pdf_bytes, file_name=f"SkinSense_Report_{patient_name}.pdf", mime="application/pdf", use_container_width=True)

st.markdown("<p style='text-align:center; color:#94A3B8; font-size:12px; margin-top:30px;'>SkinSense AI • Developed by Ahmad Raza • AI Screening Tool - Not a Final Diagnosis</p>", unsafe_allow_html=True)
