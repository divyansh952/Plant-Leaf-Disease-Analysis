# Plant Leaf Disease Detection and Visual Analysis Using Computer Vision

## 1. Project Overview

This project implements a computer vision based system for analyzing plant leaf images.

The system takes a leaf image as input, performs image preprocessing and HSV-based leaf segmentation, and uses a pretrained EfficientNet classification model to predict possible disease classes.

The system provides the top three predicted classes along with their confidence values. It also generates a segmented leaf image and a binary leaf mask for visual analysis.

The project is implemented in Python and is designed to run from the command line.

---

## 2. Objectives

The main objectives of this project are:

- To read and validate plant leaf images.
- To preprocess input images for classification.
- To segment the leaf region using computer vision techniques.
- To classify the leaf image into possible disease categories.
- To display the top three predictions with confidence values.
- To save segmentation results for visual analysis.
- To provide a command-line interface for running the system.

---

## 3. System Workflow

The project performs two main computer vision tasks:

```text
                    Input Leaf Image
                           |
                           v
                    Image Loading
                           |
                           v
                   Image Processing
                           |
              +------------+------------+
              |                         |
              v                         v
       HSV Leaf Segmentation     EfficientNet Classification
              |                         |
              v                         v
    Segmented Leaf + Mask        Top 3 Predictions
              |                         |
              +------------+------------+
                           |
                           v
                    Final Analysis
```

The segmentation and classification stages are currently performed separately.

The classifier receives the original input image rather than the segmented image.

---

## 4. Technologies Used

- Python 3
- OpenCV
- NumPy
- TensorFlow
- Keras
- Pillow
- Matplotlib
- Scikit-learn
- Hugging Face Hub

---

## 5. Project Structure

```text
Plant-Leaf-Disease-Analysis/
│
├── data/
│   └── test_leaf.jpg
│
├── models/
│   └── plantvillage/
│       └── class_names.txt
│
├── results/
│   ├── segmented_leaf.jpg
│   └── leaf_mask.jpg
│
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── classifier.py
│
├── tests/
│
├── main.py
├── requirements.txt
└── README.md
```

The pretrained `.keras` model is not stored directly in this repository because of GitHub file-size limitations.

Instead, the project automatically downloads the pretrained model from its public Hugging Face repository when the classification stage is executed.

---

## 6. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/divyansh952/Plant-Leaf-Disease-Analysis.git
```

### Step 2: Enter the project directory

```bash
cd Plant-Leaf-Disease-Analysis
```

### Step 3: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 7. Running the Project

The project can be executed from the command line.

Run:

```bash
python main.py --image data/test_leaf.jpg
```

A different image can also be provided:

```bash
python main.py --image path/to/your/image.jpg
```

During the first classification run, the pretrained EfficientNet model is downloaded automatically from Hugging Face.

The program then processes the image, displays the top three predicted classes with their confidence values, and saves the segmentation outputs.

---

## 8. Output

The program displays the top three predictions in the terminal.

Example output from the demonstration image:

```text
----------------------------------------
TOP 3 PREDICTIONS
----------------------------------------

1. Tomato___Leaf_Mold - 42.55%
2. Tomato___Septoria_leaf_spot - 28.22%
3. Tomato___Late_blight - 8.41%

----------------------------------------
```

The segmentation outputs are saved in the `results` directory:

```text
results/segmented_leaf.jpg
results/leaf_mask.jpg
```

The percentages represent the model's prediction confidence for the demonstration image. They should not be interpreted as model accuracy or as a definitive diagnosis.

---

## 9. Image Preprocessing

The input image is loaded using OpenCV.

The classification pipeline:

1. Reads the input image.
2. Converts it from BGR to RGB.
3. Resizes it to 224 × 224 pixels.
4. Converts it to a numerical representation.
5. Applies the EfficientNet preprocessing function.

The project also contains a preprocessing module for image loading, resizing, and normalization.

---

## 10. Leaf Segmentation

Leaf segmentation is performed using HSV color space thresholding.

The input image is converted from BGR to HSV.

The project uses the following HSV threshold range:

```text
Lower bound: [25, 30, 20]
Upper bound: [100, 255, 255]
```

A binary mask is generated using this range.

Morphological opening is applied to reduce small noise, while morphological closing helps fill small gaps in the detected leaf region.

The resulting mask is applied to the original image to generate the segmented leaf image.

---

## 11. Disease Classification

The classification stage uses a pretrained EfficientNet model compatible with PlantVillage disease classes.

The classifier:

1. Loads the original input image.
2. Converts it from BGR to RGB.
3. Resizes it to 224 × 224 pixels.
4. Applies EfficientNet preprocessing.
5. Runs the image through the pretrained model.
6. Selects the three classes with the highest prediction values.
7. Displays their confidence values.

The classifier currently uses the original input image. The segmented image is generated separately for visual analysis and is not passed to the classifier.

---

## 12. External Pretrained Model

The project uses the following publicly available pretrained model:

**Nefflymicn/PlantVillage-plant-disease-detection**

The model file is:

```text
plant_disease_efficientnet.keras
```

The model is downloaded automatically using the `huggingface-hub` package.

The class mapping used by the project is stored locally in:

```text
models/plantvillage/class_names.txt
```

The pretrained model is an external component and was not trained from scratch as part of this project.

---

## 13. Demonstration Image

The included demonstration image is a healthy tomato leaf image obtained from Wikimedia Commons.

Source:

https://commons.wikimedia.org/wiki/File%3AHealthy_tomato_leaves_%287871755330%29.jpg

The image is licensed under CC BY 2.0.

The image is included as a demonstration input for the project.

---

## 14. Limitations

- The classification model is based on PlantVillage disease classes.
- Performance may vary on real-world images with different lighting and backgrounds.
- The HSV-based segmentation method may be affected by background color and lighting conditions.
- Confidence values are model predictions and should not be considered a definitive diagnosis.
- The classifier currently uses the original image rather than the segmented image.
- A single demonstration image is not sufficient to evaluate overall model performance.
- The system does not provide treatment recommendations.

---

## 15. Future Scope

Possible improvements include:

- Fine-tuning the model using additional real-world leaf images.
- Improving leaf segmentation using advanced segmentation techniques.
- Using segmented leaf regions as an additional classification input.
- Adding data augmentation.
- Adding Grad-CAM for visual explanation.
- Evaluating the system using a dedicated test dataset.
- Supporting additional plant species and disease classes.

---

## 16. Conclusion

This project demonstrates a computer vision approach for plant leaf disease analysis using image preprocessing, HSV-based leaf segmentation, and pretrained deep learning classification.

The command-line system accepts a leaf image and produces the top three predicted disease classes with their confidence values. It also saves segmentation results for visual inspection.

The project demonstrates how traditional computer vision techniques and deep learning can be combined to develop an image-based plant disease analysis system.
