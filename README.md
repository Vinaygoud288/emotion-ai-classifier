# 🎭 Emotion AI — Facial Expression Classifier

A lightweight **Streamlit web app** that detects facial expressions from images using a classical machine learning model (SVM). Upload a photo, snap one with your webcam, or try a bundled sample — and the app instantly predicts the emotion displayed on the face.

---

## ✨ Features

- 📁 **Upload** a JPG/PNG image from your device
- 📷 **Camera** input — take a live photo directly in the browser
- 🖼️ **Sample gallery** — try bundled example images (place images in `samples/`)
- 🤖 **Auto face detection** — OpenCV Haar Cascade crops the face before inference
- 🔍 **Predict on demand** — click the Predict button to run inference
- 🎨 **Colour-coded result card** — unique colour and emoji per emotion
- ⚙️ **Sidebar controls** — toggle face detection on/off

## 🧠 Emotions Detected

| Emotion  | Emoji |
|----------|-------|
| Angry    | 😠    |
| Disgust  | 🤢    |
| Fear     | 😨    |
| Happy    | 😄    |
| Neutral  | 😐    |
| Sad      | 😢    |
| Surprise | 😲    |

---

## 🗂️ Project Structure

```
emotion-ai-classifier/
│
├── app.py                      # Main Streamlit application
├── emotion_model_bundle.pkl    # Trained model bundle (SVM + selector + label encoder)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Model Details

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Algorithm      | Support Vector Machine (RBF kernel)        |
| Input size     | 48 × 48 grayscale pixels                   |
| Feature selection | Variance-based selector (scikit-learn)  |
| Face detector  | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Classes        | angry, disgust, fear, happy, neutral, sad, surprise |

The model is packed into a single `emotion_model_bundle.pkl` file containing:
- `model` — trained SVM classifier
- `selector` — fitted feature selector
- `label_encoder` — maps numeric predictions back to emotion strings

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-ai-classifier.git
cd emotion-ai-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your model file

Place your trained `emotion_model_bundle.pkl` in the root of the project (alongside `app.py`).

> Model file not included due to size limitations. Train the model using the notebook or download from Google Drive.


### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📦 Requirements

```
streamlit
opencv-python-headless
numpy
joblib
Pillow
scikit-learn
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo, branch, and set `app.py` as the main file
4. Upload `emotion_model_bundle.pkl` via **Secrets** or include it via Git LFS
5. Click **Deploy** 🎉


---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) — for making ML apps incredibly easy to build
- [OpenCV](https://opencv.org/) — for face detection
- [scikit-learn](https://scikit-learn.org/) — for the SVM model and feature selection
- FER-2013 / similar datasets — commonly used for training facial expression models

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
