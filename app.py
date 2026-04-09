import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

st.set_page_config(page_title="OCR Tool", layout="centered")

st.title("⚡ OCR سريع لاستخراج النص")

uploaded_file = st.file_uploader("ارفع صورة", type=["png", "jpg", "jpeg"])

# تحميل الموديل مرة واحدة
@st.cache_resource
def load_model():
    with st.spinner("تحميل موديل OCR لأول مرة... ⏳"):
        return easyocr.Reader(['ar', 'en'], gpu=False)

reader = load_model()

# تحسين الصورة + تسريع
def preprocess(image):
    img = np.array(image)

    # تصغير الصورة لتسريع المعالجة
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # تحويل لرمادي
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # blur خفيف
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="الصورة", use_column_width=True)

    if st.button("🚀 استخراج النص"):
        with st.spinner("جاري استخراج النص... ⏳"):
            processed = preprocess(image)

            # قراءة أسرع
            results = reader.readtext(processed, detail=0, paragraph=True)

            text = "\n".join(results)

        st.success("تم استخراج النص ✅")

        st.text_area("📄 النص المستخرج", text, height=300)

        # زر تحميل
        st.download_button(
            "⬇️ تحميل النص",
            text,
            file_name="ocr_output.txt"
            )
