import streamlit as st

st.set_page_config(page_title="EmotiSign", layout="wide")

st.title("EmotiSign")
st.subheader("Real-Time Emotion and Gesture Recognition")

st.sidebar.title("Settings")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Emotion Detection", "Gesture Detection", "Both"]
)

st.write("Selected Mode:", mode)

col1, col2 = st.columns(2)

with col1:
    st.header("Webcam Feed")
    st.info("Webcam will appear here")

with col2:
    st.header("Predictions")

    st.metric("Emotion", "None")
    st.metric("Gesture", "None")

st.success("Streamlit app running successfully!")