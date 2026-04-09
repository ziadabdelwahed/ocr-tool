import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

st.title("OCR Tool")

uploaded_file = st.file_uploader("ارفع صورة", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_model():
    return easyocr.Reader(['ar', 'en'])

reader = load_model()

def preprocess(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("استخراج النص"):
        processed = preprocess(image)
        results = reader.readtext(processed, detail=0)

        text = "\n".join(results)
        st.text_area("النص", text)
