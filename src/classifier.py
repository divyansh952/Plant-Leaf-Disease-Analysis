import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download


MODEL_REPO = "Nefflymicn/PlantVillage-plant-disease-detection"
MODEL_FILENAME = "plant_disease_efficientnet.keras"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plantvillage",
    "class_names.txt"
)


def load_class_names():
    """Load PlantVillage class names."""

    class_names = {}

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if ":" in line:
                index, name = line.split(":", 1)
                class_names[int(index.strip())] = name.strip()

    return class_names


def load_disease_model():
    """Download and load the pretrained disease classification model."""

    print("Downloading/loading pretrained model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    return load_model(model_path)


def classify_image(model, image_path):
    """Predict the top three plant disease classes."""

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    predictions = model.predict(image, verbose=0)[0]

    class_names = load_class_names()

    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = []

    for index in top_indices:
        disease = class_names.get(
            int(index),
            f"Unknown class {index}"
        )

        confidence = float(predictions[index]) * 100

        top_predictions.append(
            (disease, confidence)
        )

    return top_predictions
