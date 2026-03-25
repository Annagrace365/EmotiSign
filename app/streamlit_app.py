import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

st.set_page_config(page_title="EmotiSign", layout="wide")

st.title("EmotiSign")
st.subheader("Real-Time Emotion and Gesture Recognition")

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Emotion Detection", "Gesture Detection", "Both"]
)

# Session state for sentence
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

# Dummy prediction function (REPLACE THIS)
def predict(frame):
    # 👉 Replace with your ML model
    return {
        "emotion": "Happy",
        "gesture": "Hello"
    }

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "None"
        self.gesture = "None"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = predict(img)

        self.emotion = result["emotion"]
        self.gesture = result["gesture"]

        # Optional: Draw text on frame
        cv2.putText(img, f"Emotion: {self.emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f"Gesture: {self.gesture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Webcam")

    ctx = webrtc_streamer(
        key="emotisign",
        video_processor_factory=VideoProcessor
    )

with col2:
    st.header("Predictions")

    emotion = "None"
    gesture = "None"

    if ctx.video_processor:
        emotion = ctx.video_processor.emotion
        gesture = ctx.video_processor.gesture

    # Mode-based display
    if mode == "Emotion Detection":
        st.metric("Emotion", emotion)

    elif mode == "Gesture Detection":
        st.metric("Gesture", gesture)

    else:
        st.metric("Emotion", emotion)
        st.metric("Gesture", gesture)

    # Sentence Builder
    if st.button("Add to Sentence"):
        st.session_state.sentence += gesture + " "

    if st.button("Clear Sentence"):
        st.session_state.sentence = ""

    st.write("### Sentence:")
    st.write(st.session_state.sentence)

st.success("Real-time detection running!")