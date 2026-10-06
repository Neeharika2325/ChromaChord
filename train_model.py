"""Train the AuraBeat facial-expression classifier on FER-style image folders."""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "train"
VALIDATION_DIR = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "model.h5"
EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64


def build_model() -> Sequential:
    """Create the three-block VGG-style CNN used by the detector."""
    model = Sequential(
        [
            Input(shape=(48, 48, 1)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(EMOTIONS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",   
    )
    return model


def _validate_dataset() -> Path:
    validation_dir = VALIDATION_DIR
    nested_validation_dir = TRAIN_DIR / "test"
    if not validation_dir.is_dir() and nested_validation_dir.is_dir():
        validation_dir = nested_validation_dir
        print(
            f"Using validation images from {validation_dir}; "
            f"the standard location is {VALIDATION_DIR}."
        )

    for directory in (TRAIN_DIR, validation_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}. "
                "Expected data/train and data/test, each with one folder per emotion."
            )

        class_directories = {
            child.name for child in directory.iterdir() if child.is_dir()
        }
        if directory == TRAIN_DIR and validation_dir.parent == TRAIN_DIR:
            class_directories.discard(validation_dir.name)
        if class_directories != set(EMOTIONS):
            missing = sorted(set(EMOTIONS) - class_directories)
            unexpected = sorted(class_directories - set(EMOTIONS))
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise ValueError(
                f"{directory} must contain exactly these class folders: "
                f"{', '.join(EMOTIONS)} ({'; '.join(details)})."
            )
    return validation_dir


def main() -> None:
    validation_dir = _validate_dataset()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=list(EMOTIONS),
        seed=42,
    )
    validation_generator = validation_datagen.flow_from_directory(
        str(validation_dir),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=list(EMOTIONS),
        shuffle=False,
    )
    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("Both data/train and data/test must contain image files.")

    model = build_model()
    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"Starting training; best model will be saved to {MODEL_PATH}")
    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=callbacks,
    )
    print(f"Training complete. Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
