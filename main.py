import argparse
import os
import cv2

from src.preprocessing import load_image, resize_image
from src.segmentation import segment_leaf
from src.classifier import load_disease_model, classify_image


def main():
    parser = argparse.ArgumentParser(
        description="Plant Leaf Disease Detection"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the leaf image"
    )

    args = parser.parse_args()
    image_path = args.image

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    print("\n========================================")
    print("   PLANT LEAF DISEASE DETECTION")
    print("========================================")

    print("\n[1] Loading image...")
    image = load_image(image_path)

    print("[2] Preprocessing image...")
    resize_image(image)

    print("[3] Segmenting leaf...")
    segmented_image, mask = segment_leaf(image)

    os.makedirs("results", exist_ok=True)

    cv2.imwrite(
        "results/segmented_leaf.jpg",
        segmented_image
    )

    cv2.imwrite(
        "results/leaf_mask.jpg",
        mask
    )

    print("[4] Loading classification model...")
    model = load_disease_model()

    print("[5] Classifying leaf...")

    top_predictions = classify_image(
        model,
        image_path
    )

    print("\n----------------------------------------")
    print("TOP 3 PREDICTIONS")
    print("----------------------------------------")

    for position, (disease, confidence) in enumerate(
        top_predictions,
        start=1
    ):
        print(
            f"{position}. {disease} - {confidence:.2f}%"
        )

    print("----------------------------------------")

    print("\nSegmentation result saved to:")
    print("results/segmented_leaf.jpg")

    print("Leaf mask saved to:")
    print("results/leaf_mask.jpg")

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
