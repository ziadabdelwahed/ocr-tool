"""
OCR System - Complete Implementation
Single file deployment: app.py
"""

import os
import io
import base64
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import easyocr

# ============================================
# تهيئة النظام
# ============================================

app = FastAPI(title="UCAR OCR System", version="2.0.0")

# تهيئة EasyOCR مع دعم العربية والإنجليزية
print("جار تحميل نماذج OCR...")
reader = easyocr.Reader(['ar', 'en'], gpu=False, model_storage_directory='./models')
print("النماذج جاهزة.")

# مجلدات للتخزين المؤقت
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================
# معالجة الصور المتقدمة
# ============================================

class ImagePreprocessor:
    """معالج الصور لتحسين دقة OCR"""
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """تصحيح ميل الصورة"""
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
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    @staticmethod
    def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
        """تحسين الصورة لاستخراج النص"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # إزالة التشويش
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # تطبيق threshold متكيف
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # عمليات مورفولوجية لتحسين الحروف
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def process(image_bytes: bytes) -> np.ndarray:
        """خط المعالجة الكامل"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # تصحيح الميل
        image = ImagePreprocessor.deskew(image)
        
        # تحسين الصورة
        processed = ImagePreprocessor.enhance_for_ocr(image)
        
        return processed, image

# ============================================
# معالج النصوص العربية
# ============================================

class ArabicTextProcessor:
    """معالج النصوص العربية للتنسيق والتصحيح"""
    
    # قاموس التصحيحات الشائعة
    CORRECTIONS = {
        'اللة': 'الله',
        'الرحمن': 'الرحمن',
        'اللةم': 'اللهم',
        'ة': 'ة',
        'أ': 'أ',
        'إ': 'إ',
        'آ': 'آ',
    }
    
    @staticmethod
    def fix_common_errors(text: str) -> str:
        """تصحيح الأخطاء الإملائية الشائعة في OCR"""
        for wrong, correct in ArabicTextProcessor.CORRECTIONS.items():
            text = text.replace(wrong, correct)
        
        # تصحيح المسافات
        text = re.sub(r'\s+', ' ', text)
        
        # تصحيح علامات الترقيم
        text = text.replace(' ،', '،')
        text = text.replace(' .', '.')
        text = text.replace(' ؟', '؟')
        text = text.replace(' !', '!')
        
        return text.strip()
    
    @staticmethod
    def detect_and_format_arabic(text: str) -> str:
        """تنسيق النص العربي مع مراعاة اتجاه الكتابة"""
        if not text:
            return text
            
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = len(text.strip())
        
        if total_chars > 0 and arabic_count / total_chars > 0.3:
            return ArabicTextProcessor.fix_common_errors(text)
        
        return text

# ============================================
# محرك OCR الرئيسي
# ============================================

class OCREngine:
    """محرك استخراج النصوص مع الحفاظ على التنسيق"""
    
    @staticmethod
    def extract_text_with_layout(image: np.ndarray) -> Dict[str, Any]:
        """استخراج النص مع معلومات التنسيق"""
        
        results = reader.readtext(
            image,
            detail=1,
            paragraph=True,
            width_ths=0.7,
            height_ths=0.7,
            decoder='greedy',
            beamWidth=5,
            batch_size=1,
            workers=0,
            allowlist=None,
            blocklist=None,
            rotation_info=[90, 180, 270],
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.5,
            slope_ths=0.1,
            ycenter_ths=0.5,
            height_ths=0.5,
            width_ths=0.5,
            add_margin=0.1,
        )
        
        paragraphs = []
        lines = []
        full_text_parts = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.2:
                text = ArabicTextProcessor.detect_and_format_arabic(text)
                
                line_info = {
                    'text': text,
                    'confidence': round(confidence * 100, 2),
                    'bbox': [[int(x), int(y)] for x, y in bbox],
                    'language': 'arabic' if any('\u0600' <= c <= '\u06FF' for c in text) else 'english'
                }
                lines.append(line_info)
                full_text_parts.append(text)
        
        # تجميع الفقرات
        current_paragraph = []
        for line in lines:
            current_paragraph.append(line['text'])
            if line['text'].endswith(('.', '!', '؟', '?')) or len(line['text']) < 30:
                if current_paragraph:
                    paragraphs.append({
                        'text': ' '.join(current_paragraph),
                        'lines': len(current_paragraph)
                    })
                    current_paragraph = []
        
        if current_paragraph:
            paragraphs.append({
                'text': ' '.join(current_paragraph),
                'lines': len(current_paragraph)
            })
        
        full_text = '\n'.join(full_text_parts)
        
        return {
            'full_text': full_text,
            'paragraphs': paragraphs,
            'lines': lines,
            'word_count': len(full_text.split()),
            'char_count': len(full_text),
            'confidence_avg': round(np.mean([l['confidence'] for l in lines]) if lines else 0, 2)
        }

# ============================================
# واجهة الويب HTML
# ============================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UCAR OCR - استخراج النصوص من الصور</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-panel {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 768px) {
            .main-panel {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #eef0ff;
        }
        
        .upload-area.dragover {
            border-color: #764ba2;
            background: #dde0ff;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .result-area {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .result-text {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            font-size: 16px;
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
        }
        
        .result-text[dir="rtl"] {
            text-align: right;
        }
        
        .result-text[dir="ltr"] {
            text-align: left;
        }
        
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 10px;
            color: white;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .copy-btn {
            background: #28a745;
            margin-top: 10px;
        }
        
        .copy-btn:hover {
            background: #218838;
        }
        
        .file-input {
            display: none;
        }
        
        .toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 UCAR OCR - استخراج النصوص من الصور</h1>
        
        <div class="main-panel">
            <div class="panel">
                <h2 style="margin-bottom: 20px; color: #333;">📤 رفع الصورة</h2>
                
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <h3>اسحب وأفلت الصورة هنا</h3>
                    <p style="color: #666; margin-top: 10px;">أو انقر للاختيار</p>
                    <p style="color: #999; font-size: 14px; margin-top: 15px;">
                        الصيغ المدعومة: JPG, PNG, BMP, TIFF, WEBP
                    </p>
                </div>
                
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                
                <div id="previewContainer" style="display: none;">
                    <img id="previewImage" class="preview-image" src="" alt="معاينة">
                </div>
                
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn" id="processBtn" style="display: none;">🔍 استخراج النص</button>
                    <button class="btn btn-secondary" id="clearBtn">🗑️ مسح</button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>جاري معالجة الصورة واستخراج النص...</p>
                </div>
            </div>
            
            <div class="panel">
                <h2 style="margin-bottom: 20px; color: #333;">📄 النص المستخرج</h2>
                
                <div class="stats" id="stats" style="display: none;">
                    <div class="stat-item">
                        <div class="stat-value" id="wordCount">0</div>
                        <div class="stat-label">كلمة</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="charCount">0</div>
                        <div class="stat-label">حرف</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="confidence">0%</div>
                        <div class="stat-label">الدقة</div>
                    </div>
                </div>
                
                <div class="toolbar">
                    <button class="btn copy-btn" id="copyBtn">📋 نسخ النص</button>
                    <button class="btn btn-secondary" id="downloadTxtBtn">💾 تحميل TXT</button>
                    <button class="btn btn-secondary" id="clearTextBtn">🗑️ مسح النص</button>
                </div>
                
                <div class="result-area">
                    <div class="result-text" id="resultText" dir="auto">
                        <p style="color: #999; text-align: center; padding: 40px;">
                            النص المستخرج سيظهر هنا
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const processBtn = document.getElementById('processBtn');
        const clearBtn = document.getElementById('clearBtn');
        const loading = document.getElementById('loading');
        const resultText = document.getElementById('resultText');
        const stats = document.getElementById('stats');
        const wordCount = document.getElementById('wordCount');
        const charCount = document.getElementById('charCount');
        const confidence = document.getElementById('confidence');
        const copyBtn = document.getElementById('copyBtn');
        const downloadTxtBtn = document.getElementById('downloadTxtBtn');
        const clearTextBtn = document.getElementById('clearTextBtn');
        
        let currentFile = null;
        let extractedText = '';
        
        // فتح نافذة اختيار الملف
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // السحب والإفلات
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        // اختيار ملف
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('الرجاء اختيار ملف صورة صالح');
                return;
            }
            
            currentFile = file;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';
                processBtn.style.display = 'inline-block';
                uploadArea.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        // معالجة الصورة
        processBtn.addEventListener('click', async () => {
            if (!currentFile) return;
            
            loading.style.display = 'block';
            processBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('file', currentFile);
            
            try {
                const response = await fetch('/ocr/extract', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    extractedText = data.text;
                    resultText.textContent = extractedText;
                    
                    // تحديد اتجاه النص
                    const arabicCount = (extractedText.match(/[\\u0600-\\u06FF]/g) || []).length;
                    const totalChars = extractedText.replace(/\\s/g, '').length;
                    if (totalChars > 0 && arabicCount / totalChars > 0.3) {
                        resultText.setAttribute('dir', 'rtl');
                    } else {
                        resultText.setAttribute('dir', 'ltr');
                    }
                    
                    // تحديث الإحصائيات
                    wordCount.textContent = data.word_count;
                    charCount.textContent = data.char_count;
                    confidence.textContent = data.confidence + '%';
                    stats.style.display = 'flex';
                } else {
                    alert('فشل في استخراج النص: ' + data.error);
                }
            } catch (error) {
                alert('حدث خطأ: ' + error.message);
            } finally {
                loading.style.display = 'none';
                processBtn.disabled = false;
            }
        });
        
        // نسخ النص
        copyBtn.addEventListener('click', () => {
            if (extractedText) {
                navigator.clipboard.writeText(extractedText).then(() => {
                    alert('تم نسخ النص إلى الحافظة');
                });
            }
        });
        
        // تحميل TXT
        downloadTxtBtn.addEventListener('click', () => {
            if (extractedText) {
                const blob = new Blob([extractedText], { type: 'text/plain;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'extracted_text_' + new Date().getTime() + '.txt';
                a.click();
                URL.revokeObjectURL(url);
            }
        });
        
        // مسح النص
        clearTextBtn.addEventListener('click', () => {
            resultText.textContent = 'النص المستخرج سيظهر هنا';
            resultText.style.color = '#999';
            resultText.style.textAlign = 'center';
            resultText.style.padding = '40px';
            stats.style.display = 'none';
            extractedText = '';
        });
        
        // مسح الصورة
        clearBtn.addEventListener('click', () => {
            currentFile = null;
            fileInput.value = '';
            previewContainer.style.display = 'none';
            processBtn.style.display = 'none';
            uploadArea.style.display = 'block';
        });
        
        // إعادة تعيين تنسيق النص عند وجود محتوى
        resultText.addEventListener('DOMSubtreeModified', () => {
            if (resultText.textContent && resultText.textContent !== 'النص المستخرج سيظهر هنا') {
                resultText.style.color = '#333';
                resultText.style.textAlign = resultText.getAttribute('dir') === 'rtl' ? 'right' : 'left';
                resultText.style.padding = '20px';
            }
        });
    </script>
</body>
</html>
"""

# ============================================
# مسارات API
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """الصفحة الرئيسية"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/ocr/extract")
async def extract_text(file: UploadFile = File(...)):
    """استخراج النص من الصورة"""
    try:
        # قراءة الصورة
        image_bytes = await file.read()
        
        # معالجة الصورة
        processed_image, original_image = ImagePreprocessor.process(image_bytes)
        
        # استخراج النص
        result = OCREngine.extract_text_with_layout(processed_image)
        
        if not result['full_text'].strip():
            # محاولة استخدام الصورة الأصلية
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            result = OCREngine.extract_text_with_layout(gray)
        
        return JSONResponse({
            'success': True,
            'text': result['full_text'],
            'word_count': result['word_count'],
            'char_count': result['char_count'],
            'confidence': result['confidence_avg'],
            'paragraphs': result['paragraphs'],
            'lines': result['lines']
        })
        
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)

@app.post("/ocr/extract-url")
async def extract_from_url(request: Request):
    """استخراج النص من رابط صورة"""
    try:
        data = await request.json()
        image_url = data.get('url')
        
        if not image_url:
            return JSONResponse({'success': False, 'error': 'الرابط مطلوب'}, status_code=400)
        
        import requests
        response = requests.get(image_url, timeout=30)
        image_bytes = response.content
        
        processed_image, _ = ImagePreprocessor.process(image_bytes)
        result = OCREngine.extract_text_with_layout(processed_image)
        
        return JSONResponse({
            'success': True,
            'text': result['full_text'],
            'word_count': result['word_count'],
            'char_count': result['char_count'],
            'confidence': result['confidence_avg']
        })
        
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """فحص حالة النظام"""
    return {
        'status': 'operational',
        'model': 'EasyOCR',
        'languages': ['ar', 'en'],
        'timestamp': datetime.now().isoformat()
    }

# ============================================
# نقطة البداية
# ============================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    UCAR OCR System v2.0.0                     ║
    ║                  استخراج النصوص من الصور                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  المتصفح: http://localhost:8000                              ║
    ║  API التوثيق: http://localhost:8000/docs                     ║
    ║  فحص الصحة: http://localhost:8000/health                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
        )
