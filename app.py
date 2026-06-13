import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model

MODEL_PATH = "cat_dog_cnn.h5"
IMG_SIZE = (64, 64)
# flow_from_directory assigns class indices alphabetically: cats -> 0, dogs -> 1
CLASS_NAMES = {0: "Cat", 1: "Dog"}
PREDICT_EVERY = 5  # run the model every Nth frame; reuse the label in between

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0  # match training rescale=1./255
    return np.expand_dims(arr, axis=0)


def predict(img: Image.Image):
    """Return (label_idx, confidence, raw_score)."""
    score = float(model.predict(preprocess(img), verbose=0)[0][0])  # sigmoid in [0, 1]
    label_idx = 1 if score >= 0.5 else 0
    confidence = score if label_idx == 1 else 1 - score
    return label_idx, confidence, score


def show_result(img: Image.Image):
    label_idx, confidence, score = predict(img)
    st.subheader(f"Prediction: {CLASS_NAMES[label_idx]}")
    st.progress(confidence)
    st.write(f"Confidence: **{confidence * 100:.1f}%**  (raw score: {score:.3f})")


class LiveClassifier(VideoProcessorBase):
    """Classifies the webcam stream frame-by-frame and overlays the label."""

    def __init__(self):
        self.model = get_model()
        self.frame_count = 0
        self.label = "..."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = Image.fromarray(frame.to_ndarray(format="rgb24"))

        # Throttle: predicting every frame is too slow for smooth video.
        if self.frame_count % PREDICT_EVERY == 0:
            arr = np.expand_dims(
                np.array(img.resize(IMG_SIZE), dtype="float32") / 255.0, axis=0
            )
            score = float(self.model.predict(arr, verbose=0)[0][0])
            idx = 1 if score >= 0.5 else 0
            conf = score if idx == 1 else 1 - score
            self.label = f"{CLASS_NAMES[idx]} ({conf * 100:.0f}%)"
        self.frame_count += 1

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 28], fill=(0, 0, 0))
        draw.text((6, 6), self.label, fill=(0, 255, 0))

        return av.VideoFrame.from_ndarray(np.array(img), format="rgb24")


st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Run **live** prediction, snap a **camera** photo, or **upload** an image.")

model = get_model()

source = st.radio(
    "Input source", ["🔴 Live", "📷 Camera", "📁 Upload"], horizontal=True
)

if source == "🔴 Live":
    st.caption(
        f"The CNN predicts continuously (every {PREDICT_EVERY} frames) and overlays "
        "the label on the video."
    )
    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveClassifier,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif source == "📷 Camera":
    shot = st.camera_input("Take a picture")
    if shot is not None:
        show_result(Image.open(shot))

else:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Predict"):
            show_result(img)
