# Handwritten Digit Recognition Using CNN

## Project Overview

This project is a Handwritten Digit Recognition system developed using Deep Learning (Convolutional Neural Networks). It recognizes handwritten digits (0–9) from uploaded images using the MNIST dataset.

## Features

- Handwritten digit recognition
- Image preprocessing using OpenCV
- CNN model built using TensorFlow/Keras
- Upload custom handwritten images
- Predict digit with confidence score
- Displays processed image before prediction

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit

## Dataset

- MNIST Dataset
- 60,000 Training Images
- 10,000 Testing Images
- Image Size: 28 × 28 pixels

## Model Architecture

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten
- Dense (128 Neurons)
- Dense (10 Neurons - Softmax)

## Workflow

1. Load MNIST Dataset
2. Preprocess Images
3. Train CNN Model
4. Evaluate Model
5. Save Model
6. Upload Custom Image
7. Predict Handwritten Digit

## Accuracy

- Training Accuracy: ~99%
- Test Accuracy: ~99%

## Project Structure

```
Handwritten-Character-Recognition/
│
├── app.py
├── cnn_model.keras
├── requirements.txt
├── README.md
└── sample_images/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Future Enhancements

- Handwritten alphabet recognition
- Word recognition
- Real-time webcam prediction
- Mobile application

## Author

Vethaabinaya S
