import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

st.title("OCR Tool - Custom")

uploaded_file = st.file_uploader("ارفع صورة", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_model():
    return easyocr.Reader(['ar', 'en'], gpu=False)

reader = load_model()

# ✅ preprocess (مظبوط)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    coords = np.column_stack(np.where(denoised > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    (h, w) = denoised.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(denoised, M, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(rotated)

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return binary


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="الصورة", use_column_width=True)

    if st.button("🚀 استخراج النص"):
        with st.spinner("جاري المعالجة..."):
            processed = preprocess_image(cv_image)

            results = reader.readtext(processed, detail=0, paragraph=True)

            text = "\n".join(results)

        st.success("تم استخراج النص ✅")
        st.text_area("النص", text, height=300)
