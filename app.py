"""Streamlit interface for live emotion detection and mood-based music."""

import atexit
from pathlib import Path
import urllib.parse

import cv2
import pandas as pd
import streamlit as st

from emotion_detector import EMOTIONS, EmotionDetector
from mood_mapper import get_search_query

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
_open_captures: list[cv2.VideoCapture] = []


def _release_capture(capture: cv2.VideoCapture | None) -> None:
    if capture is not None:
        capture.release()
        if capture in _open_captures:
            _open_captures.remove(capture)


def _release_all_captures() -> None:
    for capture in _open_captures:
        capture.release()
    _open_captures.clear()


def _stop_camera() -> None:
    _release_capture(st.session_state.get("camera_capture"))
    st.session_state.camera_capture = None
    st.session_state.camera_enabled = False


atexit.register(_release_all_captures)

st.set_page_config(page_title="AuraBeat", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
      .main { background: linear-gradient(145deg, #0b1020 0%, #12182a 100%); }
      .block-container { padding-top: 2rem; }
      .aura-subtitle { color: #aab4ca; font-size: 1.05rem; }
      .mood-badge {
        display: inline-block; padding: .45rem .9rem; border-radius: 999px;
        background: #32255e; color: #e5d9ff; font-weight: 700;
        letter-spacing: .04em; text-transform: capitalize;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎧 AuraBeat")
st.markdown(
    '<p class="aura-subtitle">Let your mood shape the soundtrack.</p>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(
        f"The trained model is missing: `{MODEL_PATH}`. "
        "Prepare both dataset splits and run `python train_model.py` first."
    )
    st.stop()
if not CASCADE_PATH.is_file():
    st.error(
        f"OpenCV's Haar cascade is missing: `{CASCADE_PATH}`. "
        "Place `haarcascade_frontalface_default.xml` next to the app scripts."
    )
    st.stop()

if "camera_capture" not in st.session_state:
    st.session_state.camera_capture = None
if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = False
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None
if "emotion_probabilities" not in st.session_state:
    st.session_state.emotion_probabilities = None
if "last_played_track" not in st.session_state:
    st.session_state.last_played_track = None
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = None


@st.cache_resource
def load_detector() -> EmotionDetector:
    return EmotionDetector()


with st.sidebar:
    st.header("Camera")
    camera_enabled = st.toggle("Start webcam", key="camera_enabled")
    st.button("Stop webcam", on_click=_stop_camera, disabled=not camera_enabled)
    st.caption("Your camera frames are processed locally by the emotion model.")


@st.fragment(run_every=0.25)
def render_dashboard(active: bool, model: EmotionDetector) -> None:
    left_column, right_column = st.columns([1.15, 0.85], gap="large")

    with left_column:
        st.subheader("Live camera")
        if active:
            capture = st.session_state.camera_capture
            if capture is None or not capture.isOpened():
                if capture is not None:
                    _release_capture(capture)
                capture = cv2.VideoCapture(0)
                _open_captures.append(capture)
                st.session_state.camera_capture = capture

            if not capture.isOpened():
                _release_capture(capture)
                st.session_state.camera_capture = None
                st.error(
                    "Unable to access the webcam. Check camera permissions and "
                    "that another application is not using it."
                )
            else:
                success, frame = capture.read()
                if not success or frame is None:
                    st.error("The webcam is connected but did not return a frame.")
                else:
                    emotion, annotated_frame, probabilities = model.detect_emotion(frame)
                    st.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    st.session_state.detected_emotion = emotion
                    st.session_state.emotion_probabilities = probabilities
                    if emotion is None:
                        st.info("No face detected. Center your face in the camera.")
        else:
            capture = st.session_state.camera_capture
            if capture is not None:
                _release_capture(capture)
                st.session_state.camera_capture = None
            st.session_state.detected_emotion = None
            st.session_state.emotion_probabilities = None
            st.info("Turn on **Start webcam** in the sidebar to begin.")

    with right_column:
        st.subheader("Your current mood")
        emotion = st.session_state.detected_emotion
        probabilities = st.session_state.emotion_probabilities
        if emotion and probabilities:
            st.markdown(
                f'<span class="mood-badge">{emotion}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.caption("Emotion confidence")
            score_table = pd.DataFrame(
                {
                    "Emotion": [label.title() for label in EMOTIONS],
                    "Probability": [
                        float(probabilities[label]) for label in EMOTIONS
                    ],
                }
            )
            st.dataframe(
                score_table.style.format({"Probability": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
            for row in score_table.itertuples(index=False):
                st.write(f"{row.Emotion} · {row.Probability:.1%}")
                st.progress(min(max(float(row.Probability), 0.0), 1.0))

            if st.button("Play Mood Track", type="primary", use_container_width=True):
                query = get_search_query(emotion)
                encoded_query = urllib.parse.quote_plus(query)
                yt_link = f"https://www.youtube.com/results?search_query={encoded_query}"
                
                st.session_state.last_played_track = query
                st.session_state.youtube_url = yt_link
                st.rerun(scope="app")
        else:
            st.info("Your mood and emotion probabilities will appear here.")
            st.button(
                "Play Mood Track",
                type="primary",
                use_container_width=True,
                disabled=True,
            )
        if st.session_state.last_played_track:
            st.caption(f"Last selected mood: {st.session_state.last_played_track}")


try:
    detector = load_detector()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Could not initialize emotion detection: {error}")
    st.stop()

render_dashboard(camera_enabled, detector)

if st.session_state.youtube_url:
    st.markdown("---")
    st.subheader("🎧 Recommended Soundtrack")
    st.write(f"Generated playlist for mood: **{st.session_state.last_played_track}**")
    st.link_button(
        f"▶️ Listen to '{st.session_state.last_played_track}' on YouTube",
        st.session_state.youtube_url,
        type="primary",
        use_container_width=True,
    )