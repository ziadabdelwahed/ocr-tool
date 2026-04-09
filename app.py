import streamlit as st

# لازم أول حاجة
st.set_page_config(page_title="UCAR OCR", layout="wide")

import os
import re
import cv2
import numpy as np
from PIL import Image
import easyocr

st.title("🔍 UCAR OCR System")

# ============================================
# تحميل الموديل
# ============================================

@st.cache_resource
def load_model():
    return easyocr.Reader(['ar', 'en'], gpu=False)

reader = load_model()

# ============================================
# معالجة الصور
# ============================================

class ImagePreprocessor:

    @staticmethod
    def deskew(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        coords = np.column_stack(np.where(gray > 0))

        if len(coords) < 100:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.5:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def enhance_for_ocr(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary

    @staticmethod
    def process(image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image = ImagePreprocessor.deskew(image)
        processed = ImagePreprocessor.enhance_for_ocr(image)

        return processed, image

# ============================================
# معالجة النص العربي
# ============================================

class ArabicTextProcessor:

    CORRECTIONS = {
        'اللة': 'الله',
        'اللةم': 'اللهم',
    }

    @staticmethod
    def fix_common_errors(text):
        for wrong, correct in ArabicTextProcessor.CORRECTIONS.items():
            text = text.replace(wrong, correct)

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def detect_and_format_arabic(text):
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total = len(text.strip())

        if total > 0 and arabic_count / total > 0.3:
            return ArabicTextProcessor.fix_common_errors(text)

        return text

# ============================================
# OCR Engine
# ============================================

class OCREngine:

    @staticmethod
    def extract_text_with_layout(image):

        results = reader.readtext(
            image,
            detail=1,
            paragraph=True,
            height_ths=0.5,
            width_ths=0.5,
        )

        lines = []
        full_text_parts = []

        for (bbox, text, confidence) in results:
            if confidence > 0.2:
                text = ArabicTextProcessor.detect_and_format_arabic(text)

                lines.append({
                    'text': text,
                    'confidence': round(confidence * 100, 2)
                })

                full_text_parts.append(text)

        full_text = "\n".join(full_text_parts)

        return {
            'full_text': full_text,
            'word_count': len(full_text.split()),
            'char_count': len(full_text),
            'confidence_avg': round(
                np.mean([l['confidence'] for l in lines]) if lines else 0, 2
            )
        }

# ============================================
# واجهة المستخدم
# ============================================

uploaded_file = st.file_uploader("📤 ارفع صورة", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة", use_column_width=True)

    if st.button("🚀 استخراج النص"):

        with st.spinner("جاري المعالجة... ⏳"):

            image_bytes = uploaded_file.read()

            processed, original = ImagePreprocessor.process(image_bytes)

            result = OCREngine.extract_text_with_layout(processed)

            # fallback
            if not result['full_text'].strip():
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                result = OCREngine.extract_text_with_layout(gray)

        st.success("تم استخراج النص ✅")

        st.text_area("📄 النص", result['full_text'], height=300)

        col1, col2, col3 = st.columns(3)

        col1.metric("كلمات", result['word_count'])
        col2.metric("حروف", result['char_count'])
        col3.metric("دقة", f"{result['confidence_avg']}%")

        st.download_button(
            "⬇️ تحميل النص",
            result['full_text'],
            file_name="ocr.txt"
    )
