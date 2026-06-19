import streamlit as st
import cv2
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Emotion AI | Facial Expression Classifier",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — professional look
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid #e0e0e0;
    }
    .result-emoji {
        font-size: 4rem;
        margin-bottom: 0.3rem;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: capitalize;
        margin: 0;
        color: #111111 !important;
    }
    .footer-note {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        padding-top: 2rem;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
    }
    /* Make predict button stand out */
    div[data-testid="stButton"] > button[kind="primary"] {
        width: 100%;
        font-size: 1.05rem;
        padding: 0.65rem 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Constants
# ============================================================
MODEL_PATH = "emotion_model_bundle.pkl"
SAMPLES_DIR = "samples"  # put a few example .jpg/.png face images here to enable the "Try a Sample" tab

EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "angry": "#e74c3c",
    "disgust": "#27ae60",
    "fear": "#8e44ad",
    "happy": "#f1c40f",
    "neutral": "#95a5a6",
    "sad": "#3498db",
    "surprise": "#e67e22",
}

# ============================================================
# Cached resource loaders
# ============================================================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    artifacts = joblib.load(MODEL_PATH)
    return artifacts["model"], artifacts["selector"], artifacts["label_encoder"]

@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

@st.cache_data
def list_sample_images():
    """Look for any sample images bundled in a local 'samples/' folder."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted([f for f in Path(SAMPLES_DIR).iterdir() if f.suffix.lower() in exts])
    return files

model, selector, le = load_artifacts()
face_cascade = load_face_detector()
sample_files = list_sample_images()

# ============================================================
# Core inference functions
# ============================================================
def detect_and_crop_face(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return gray_img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_img[y:y + h, x:x + w], True

def predict_emotion(gray_img):
    resized = cv2.resize(gray_img, (48, 48))
    flat = resized.flatten().reshape(1, -1).astype(np.float32) / 255.0
    flat_selected = selector.transform(flat)
    pred = model.predict(flat_selected)
    emotion = le.inverse_transform(pred)[0]
    return emotion, resized

def run_pipeline(pil_image, use_face_detection):
    gray_array = np.array(pil_image.convert("L"))
    face_found = False
    if use_face_detection:
        face_img, face_found = detect_and_crop_face(gray_array)
    else:
        face_img = gray_array
    emotion, model_input = predict_emotion(face_img)
    return emotion, model_input, face_found

def render_result(emotion, model_input, face_found, use_face_detection, original_image):
    emoji = EMOTION_EMOJI.get(emotion, "🙂")
    color = EMOTION_COLOR.get(emotion, "#333")

    st.markdown(f"""
    <div class="result-card" style="border-top: 4px solid {color};">
        <div class="result-emoji">{emoji}</div>
        <p class="result-label">{emotion}</p>
    </div>
    """, unsafe_allow_html=True)

    if use_face_detection:
        if face_found:
            st.success("Face automatically detected and cropped before prediction.")
        else:
            st.warning("No face detected — prediction was made on the full image, which may reduce accuracy.")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_face_detection = st.toggle("Auto-detect & crop face", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption("**Algorithm:** SVM (RBF kernel)")
    st.caption("**Input size:** 48 × 48 grayscale")
    st.caption("**Classes:** angry, disgust, fear, happy, neutral, sad, surprise")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This app classifies facial expressions from images using a classical "
        "machine learning model trained on grayscale facial image data."
    )

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎭 Emotion AI</h1>
    <p>Upload a face, and let the model read the expression.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Guard: model missing
# ============================================================
if model is None:
    st.error(
        f"Model file `{MODEL_PATH}` was not found in the app directory. "
        "Please make sure it's placed alongside `app.py` before running."
    )
    st.stop()

# ============================================================
# Tabs: Upload / Camera / Samples
# ============================================================
tab_labels = ["📁 Upload", "📷 Camera"]
if sample_files:
    tab_labels.append("🖼️ Try a Sample")

tabs = st.tabs(tab_labels)

selected_image = None

# --- Upload tab ---
with tabs[0]:
    uploaded_file = st.file_uploader("Upload a clear, front-facing face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
        except Exception:
            st.error("Couldn't read that file. Please upload a valid JPG or PNG image.")

# --- Camera tab ---
with tabs[1]:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        selected_image = Image.open(camera_file)

# --- Sample gallery tab (only if samples/ folder has images) ---
if sample_files:
    with tabs[2]:
        st.write("Pick a sample image to see the model in action:")
        cols = st.columns(min(len(sample_files), 4))
        chosen_path = None
        for i, path in enumerate(sample_files):
            with cols[i % len(cols)]:
                thumb = Image.open(path)
                st.image(thumb, use_container_width=True)
                if st.button(path.stem.replace("_", " ").title(), key=f"sample_{i}", use_container_width=True):
                    chosen_path = path
        if chosen_path is not None:
            selected_image = Image.open(chosen_path)

# ============================================================
# Preview + Predict button
# ============================================================
if selected_image is not None:
    st.image(selected_image, caption="Selected image — ready to analyse", use_container_width=True)

    predict_clicked = st.button("🔍 Predict Emotion", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analysing facial expression..."):
            try:
                emotion, model_input, face_found = run_pipeline(selected_image, use_face_detection)
                render_result(emotion, model_input, face_found, use_face_detection, selected_image)
            except Exception as e:
                st.error(f"Something went wrong while processing the image: {e}")
else:
    st.info("👆 Upload an image, take a photo, or try a sample to get started.")

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer-note">
    Built with Streamlit · SVM model trained on grayscale facial data ·
    Results may vary with lighting, angle, and image quality.
</div>
""", unsafe_allow_html=True)