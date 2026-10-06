"""Face detection and seven-class facial-expression inference."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tensorflow.keras.models import load_model


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


class EmotionDetector:
    """Load the trained model and infer an emotion from the largest face."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        model_file = base_dir / "model.keras"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {model_file}. "
                "Run train_model.py after preparing data/train and data/test."
            )

        cascade_file = base_dir / "haarcascade_frontalface_default.xml"
        if not cascade_file.is_file():
            raise FileNotFoundError(
                f"OpenCV Haar cascade not found at {cascade_file}."
            )

        if not hasattr(cv2, "CascadeClassifier"):
            version = getattr(cv2, "__version__", "unknown")
            raise RuntimeError(
                f"OpenCV {version} does not provide cv2.CascadeClassifier. "
                "Install a compatible OpenCV 4.x build with "
                "`python -m pip install --force-reinstall "
                "\"opencv-python>=4.8,<5\"`."
            )

        self.face_cascade = cv2.CascadeClassifier(str(cascade_file))
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_file}")

        self.model = load_model(str(model_file))
        output_shape = self.model.output_shape
        if isinstance(output_shape, list) or output_shape[-1] != len(EMOTIONS):
            raise ValueError(
                f"Expected a model with {len(EMOTIONS)} emotion outputs; "
                f"found output shape {output_shape}."
            )

    def detect_emotion(
        self, frame: np.ndarray
    ) -> tuple[Optional[str], np.ndarray, Optional[dict[str, float]]]:
        """Return the strongest face's emotion, an annotated frame, and scores."""
        if frame is None or frame.size == 0:
            raise ValueError("The camera returned an empty frame.")
        if frame.ndim == 2:
            grayscale = frame
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated_frame = frame.copy()
        else:
            raise ValueError("Expected a grayscale or three-channel BGR frame.")

        faces = self.face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return None, frame, None

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        face_roi = grayscale[y : y + height, x : x + width]
        resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        model_input = resized.astype(np.float32).reshape(1, 48, 48, 1) / 255.0

        predictions = np.asarray(self.model.predict(model_input, verbose=0))
        if predictions.shape != (1, len(EMOTIONS)):
            raise RuntimeError(
                f"Model returned unexpected prediction shape {predictions.shape}."
            )
        scores = predictions[0]
        emotion_index = int(np.argmax(scores))
        emotion = EMOTIONS[emotion_index]
        probabilities = {
            label: float(score) for label, score in zip(EMOTIONS, scores)
        }

        color = (40, 210, 120)
        cv2.rectangle(
            annotated_frame, (x, y), (x + width, y + height), color, thickness=2
        )
        cv2.putText(
            annotated_frame,
            f"{emotion.title()} {probabilities[emotion]:.0%}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return emotion, annotated_frame, probabilities