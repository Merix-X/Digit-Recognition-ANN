# Digit-Recognition-ANN

A simple Python project using Keras to train and run a neural network that recognizes handwritten digits from 28x28 pixel images.

## 🧠 Overview

This repository contains two main scripts:

- `train_model.py` — trains a Keras model to recognize digits using the MNIST dataset.
- `main.py` — loads the trained model and predicts the digit from a user-provided image.

The model is saved as `digit_recognition_model.keras` and is already included in this repository — so you're ready to start predicting right away!

## 📦 Clone the Repository

```bash
git clone https://github.com/Merix-X/Digit-Recognition-ANN.git
cd Digit-Recognition-ANN
```

## ⚙️ Installation

Before running any scripts, install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ Train Your Own Model — or Use the Included One

Want to train the model yourself? Just run:

```bash
python train_model.py
```

This will use the MNIST dataset to train a fresh model and save it as `digit_recognition_model.keras`.

Prefer to skip training? No problem — the repository already includes a pre-trained model file, so you can jump straight to predictions.

## 🔍 Run Prediction

Once the model is available (either pre-trained or freshly created), you can run predictions:

```bash
python main.py
```

You will be prompted to enter the path to a 28x28 pixel image containing a handwritten digit. The image must:

- Be in grayscale or RGB
- Have a white background
- Contain a black digit

You can use the included `digit.png` file (a sample digit "8") to test the model.

## 📊 Output

The model will display a bar chart showing its confidence for each digit (0–9). Each bar represents the probability that the input image corresponds to that digit.

## ✍️ Tips for Drawing Digits

For best results:

- Draw the digit centered in the image
- Use a pen thickness of ~3 pixels
- Avoid making the digit too large or touching the edges
- Ensure high contrast: black digit on white background

## 👤 Author

This repository and README were created by **Merix-X**.
