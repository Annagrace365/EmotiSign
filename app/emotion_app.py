import streamlit as st
import cv2
import numpy as np
import os
import av

from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join("model", "emotion_model.h5")

model = load_model(MODEL_PATH)

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# PREPROCESS
# =========================
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

# =========================
# PREDICT
# =========================
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        processed = preprocess_face(face)
        preds = model.predict(processed)

        label = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)

        results.append((x, y, w, h, label, confidence))

    return results

# =========================
# STREAMLIT UI
# =========================
st.title("Real-Time Emotion Detection")

# =========================
# VIDEO PROCESSOR
# =========================
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = detect_emotion(img)

        for (x, y, w, h, label, conf) in results:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(img,
                        f"{label} ({conf:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# START CAMERA
# =========================
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor
)