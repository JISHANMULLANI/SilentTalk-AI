
import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from src.speech import speak_text

from src.camera import Camera
from src.hand_detection import HandDetector
from src.preprocessing import (
    extract_hand_landmarks,
    normalize_landmarks,
)
from src.prediction import SignPredictor


# -------------------- PATHS --------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "sign_lstm.keras"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.jpg"

SEQUENCE_LENGTH = 30


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SignSpeak AI | Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- CUSTOM DESIGN --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #202044 0%, transparent 38%),
            #0b0d16;
        color: #f4f5ff;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        font-family: 'Inter', sans-serif !important;
    }

    .hero {
        background: linear-gradient(120deg, #1a1c35, #171329 60%, #25204a);
        border: 1px solid #383452;
        border-radius: 24px;
        padding: 26px 30px;
        margin: 8px 0 25px 0;
    }

    .eyebrow {
        color: #c4b5fd;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 44px);
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 8px 0;
        color: #ffffff;
    }

    .hero-description {
        color: #b8bdd5;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .panel {
        background: #141725;
        border: 1px solid #292e43;
        border-radius: 20px;
        padding: 20px;
        min-height: 130px;
    }

    .panel-heading {
        color: #f4f5ff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .panel-subtitle {
        color: #929bb7;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .letter-card {
        background: linear-gradient(145deg, #221b42, #171a2c);
        border: 1px solid #6d4bb5;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }

    .letter-label {
        color: #c4b5fd;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 700;
    }

    .letter-value {
        font-size: 90px;
        line-height: 1.2;
        font-weight: 800;
        color: #ffffff;
        margin: 8px 0;
    }

    .letter-note {
        color: #b7b8d0;
        font-size: 13px;
    }

    .feature {
        background: #141725;
        border: 1px solid #292e43;
        border-radius: 16px;
        padding: 18px;
        min-height: 110px;
    }

    .feature-title {
        color: #eeeaff;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .feature-text {
        color: #9ba4c0;
        font-size: 13px;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #9a78ff;
        background: linear-gradient(100deg, #7953e8, #a24be7);
        color: white;
        font-weight: 700;
        min-height: 45px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #c4b5fd;
        color: white;
        filter: brightness(1.12);
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    hr {
        border-color: #292e43;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- HEADER --------------------
header_left, header_right = st.columns([1, 5], vertical_alignment="center")

with header_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=105)
    else:
        st.markdown("## 🤟")

with header_right:
    st.markdown(
        """
        <div>
            <div style="font-size:27px;font-weight:800;color:#fff;">
                SignSpeak <span style="color:#b69aff;">AI</span>
            </div>
            <div style="font-size:13px;color:#aab1cc;margin-top:4px;">
                REAL-TIME SIGN LANGUAGE RECOGNITION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Breaking communication barriers</div>
        <div class="hero-title">Your hands. Your voice. Your world.</div>
        <p class="hero-description">
            Recognize trained sign-language alphabet gestures in real time
            using computer vision and an LSTM deep-learning model.
            Show a sign to your camera and see the predicted letter.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- CHECK FILES --------------------
if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    st.error(
        "Model files not found. Check models/sign_lstm.keras "
        "and models/label_encoder.pkl."
    )
    st.stop()


# -------------------- SESSION STATE --------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "last_letter" not in st.session_state:
    st.session_state.last_letter = "—"

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0


# -------------------- MAIN LAYOUT --------------------
camera_col, result_col = st.columns([1.65, 1], gap="large")

with camera_col:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">📷 Live camera</div>
            <div class="panel-subtitle">
                Position your hand clearly inside the camera view.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with result_col:
    st.markdown(
        """
        <div class="panel-heading">✨ Recognition result</div>
        <div class="panel-subtitle">
            Your currently recognized alphabet letter.
        </div>
        """,
        unsafe_allow_html=True,
    )

    letter_placeholder = st.empty()
    confidence_placeholder = st.empty()
    help_placeholder = st.empty()

    st.markdown(
        """
        <div class="panel" style="margin-top:15px;">
            <div class="panel-heading">💡 Tips for better results</div>
            <p class="feature-text">
                • Keep your hand fully visible.<br>
                • Use a plain background and good lighting.<br>
                • Hold a trained sign steadily for a moment.<br>
                • Try signs included in your training dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- ACTION BUTTON --------------------
button_col, info_col = st.columns([1, 2])

with button_col:
    button_label = (
        "⏹ Stop recognition"
        if st.session_state.running
        else "▶ Start recognition"
    )

    if st.button(button_label, use_container_width=True):
        st.session_state.running = not st.session_state.running

with info_col:
    st.caption(
        "Model: LSTM  •  Features: 63 hand coordinates  •  "
        "Sequence: 30 frames"
    )


def show_letter(letter, confidence):
    st.session_state.last_letter = letter
    st.session_state.last_confidence = confidence

    shown_letter = letter if letter != "?" else "?"

    letter_placeholder.markdown(
        f"""
        <div class="letter-card">
            <div class="letter-label">Predicted letter</div>
            <div class="letter-value">{shown_letter}</div>
            <div class="letter-note">
                {"Gesture recognized" if letter != "?" else "Hold a trained sign steadily"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence_placeholder.progress(
        max(0.0, min(1.0, confidence)),
        text=f"Model confidence: {confidence:.1%}",
    )


# Show the last result when recognition is stopped.
if not st.session_state.running:
    show_letter(
        st.session_state.last_letter,
        st.session_state.last_confidence,
    )
    status_placeholder.info("Camera is stopped. Start recognition when ready.")


# -------------------- REAL-TIME RECOGNITION --------------------
if st.session_state.running:
    camera = None
    detector = None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        # Only load a pickle generated by your own trusted training script.
        with open(ENCODER_PATH, "rb") as file:
            label_encoder = pickle.load(file)

        predictor = SignPredictor(model, label_encoder)

        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        prediction_history = deque(maxlen=5)
        last_spoken_letter = None  # Avoid repeating speech for a held gesture

        camera = Camera()
        detector = HandDetector(max_num_hands=1)

        status_placeholder.success("Camera active — show a trained sign.")

        while st.session_state.running:
            frame = camera.read_frame()

            if frame is None:
                status_placeholder.error("Unable to read the webcam.")
                break

            frame, results = detector.detect_hands(frame)

            if results.multi_hand_landmarks:
                features = extract_hand_landmarks(results)
                features = normalize_landmarks(features)
                frame_buffer.append(features)

                if len(frame_buffer) == SEQUENCE_LENGTH:
                    sequence = np.stack(frame_buffer).astype(np.float32)
                    letter, confidence = predictor.predict(sequence)

                    prediction_history.append(letter)

                    valid = [p for p in prediction_history if p != "?"]

                    display_letter = "?"
                    if valid:
                        counts = {
                            p: valid.count(p)
                            for p in set(valid)
                        }
                        best = max(counts, key=counts.get)

                        if counts[best] >= 3:
                            display_letter = best

                    cv2.putText(
                        frame,
                        f"Letter: {display_letter}",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 255, 100) if display_letter != "?"
                        else (0, 165, 255),
                        3,
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.0%}",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                    show_letter(display_letter, confidence)

                    # Speak each stable letter once.
                    # Remove your hand and show the gesture again to repeat it.
                    if display_letter != "?" and display_letter != last_spoken_letter:
                        try:
                            speak_text(display_letter)
                            last_spoken_letter = display_letter
                            status_placeholder.success(
                                f"Recognized and spoken: {display_letter}"
                            )
                        except Exception as speech_error:
                            status_placeholder.error(
                                f"Speech error: {speech_error}"
                            )

                else:
                    help_placeholder.caption(
                        f"Preparing sequence: {len(frame_buffer)}/{SEQUENCE_LENGTH} frames"
                    )

            else:
                frame_buffer.clear()
                prediction_history.clear()
                last_spoken_letter = None
                show_letter("?", 0.0)
                status_placeholder.warning(
                    "No hand detected. Show one hand to the camera."
                )

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

    except Exception as exc:
        status_placeholder.error(f"Recognition error: {exc}")
        st.exception(exc)

    finally:
        if detector is not None:
            detector.close()

        if camera is not None:
            camera.release()

        st.session_state.running = False



# -------------------- TEXT AND SPEECH --------------------

if "constructed_text" not in st.session_state:
    st.session_state.constructed_text = ""

st.markdown("---")
st.markdown("## 💬 Text & Speech")
st.caption(
    "Add recognized letters to construct text, then speak it aloud."
)

st.text_area(
    "Constructed text",
    key="constructed_text",
    height=120,
    placeholder="Your letters will build text here...",
)

col_add, col_space, col_delete, col_clear, col_speak = st.columns(5)

with col_add:
    manual_letter = st.selectbox(
        "Letter",
        [chr(code) for code in range(ord("A"), ord("Z") + 1)],
        label_visibility="collapsed",
    )
    if st.button("Add letter", use_container_width=True):
        st.session_state.constructed_text += manual_letter
        st.rerun()

with col_space:
    if st.button("Space", use_container_width=True):
        if st.session_state.constructed_text and not st.session_state.constructed_text.endswith(" "):
            st.session_state.constructed_text += " "
        st.rerun()

with col_delete:
    if st.button("⌫ Delete", use_container_width=True):
        st.session_state.constructed_text = st.session_state.constructed_text[:-1]
        st.rerun()

with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.constructed_text = ""
        st.rerun()

with col_speak:
    if st.button("🔊 Speak", use_container_width=True):
        if st.session_state.constructed_text.strip():
            try:
                if speak_text(st.session_state.constructed_text):
                    st.success("Speech completed.")
            except Exception as exc:
                st.error(f"Speech error: {exc}")
        else:
            st.warning("Add some text before speaking.")
