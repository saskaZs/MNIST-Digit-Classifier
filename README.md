# ✍️ Handwritten Digit Recognizer

A simple yet effective neural network model for recognizing handwritten digits (0-9), trained on the MNIST-like dataset using TensorFlow/Keras. This project demonstrates end-to-end machine learning: data loading, preprocessing, model building, training, evaluation, and inference on custom images.

The model uses a convolutional neural network (CNN) to achieve high accuracy (~98%+ on validation). Includes visualization of training history and image preprocessing for real-world inputs.

---

## 🚀 Features

- **Data Handling:** Loads CSV data (e.g., Kaggle's Digit Recognizer), normalizes, reshapes into images.
- **CNN Model:** Sequential Keras model with Conv2D, MaxPooling, Dropout for regularization, and Dense layers.
- **Training:** Splits data, trains with Adam optimizer, categorical crossentropy; saves model as .keras.
- **Evaluation:** Validation accuracy/loss, plus matplotlib plots for history.
- **Inference:** Preprocesses custom PNG images (crop, resize, invert if needed), predicts digit with confidence.
- **Customization:** Easy config for image size, classes, test split, epochs.
- **Pure Python:** No external APIs; runs locally with standard ML libs.

---

## 📂 Project Structure
`main.py`                     # Main script: Load data, build/train/eval/save model, plot history
`predict.py`                  # Inference script: Load model, preprocess custom image, predict digit
`requirements.txt`            # Dependencies: pandas, tensorflow, scikit-learn, matplotlib, pillow, numpy


**Note:** Assumes `train.csv` (from Kaggle) in root for training, and a user-provided PNG (e.g., `my_digit.png`) for prediction.

---

## 🧠 Theoretical Background

This project implements a classic **computer vision task**: handwritten digit recognition, inspired by the iconic **MNIST dataset** (Modified National Institute of Standards and Technology).

### 1. The Problem & Dataset
- **Task**: Classify grayscale 28x28 pixel images of digits (0-9) into 10 classes.
- **MNIST-like Data**: Kaggle's Digit Recognizer provides 42,000 training samples in CSV format (label + 784 pixels).
- **Challenges**: Variations in handwriting style, thickness, slant, noise — requires robust feature extraction.

### 2. Preprocessing
- **Normalization**: Pixels (0-255) → [0,1] to stabilize gradients during training.
- **Reshaping**: Flat 784-vector → (28,28,1) image tensor for CNN input.
- **Train/Val Split**: 80/20 random split to evaluate generalization (avoid overfitting).
- **Inference Prep**: For custom images:
  - Convert to grayscale, invert if light background.
  - Crop to bounding box (remove empty space).
  - Resize to ~20px (MNIST scale) while preserving aspect.
  - Center in 28x28 black canvas.

### 3. Convolutional Neural Network (CNN) Architecture
CNNs excel at image tasks by learning hierarchical features (edges → shapes → digits).

Model structure:
- **Input**: (28,28,1) grayscale image.
- **Conv Layers**: 2x Conv2D (32/64 filters, 3x3 kernel, ReLU) to detect local patterns.
- **Pooling**: MaxPooling2D (2x2) reduces dimensions, adds translation invariance.
- **Dropout**: 25% rate prevents overfitting by random neuron deactivation.
- **Flatten + Dense**: Flatten to 1D, then Dense(128, ReLU) → Softmax(10) for classification.

**Why CNN over MLP?**
- **Parameter efficiency**: Shared weights in convolutions → fewer params than fully connected.
- **Spatial hierarchy**: Early layers learn low-level features (edges), later high-level (digit shapes).
- **Formula for Conv**: Output size = ((W - K + 2P)/S) + 1
  - W: Input width, K: Kernel size, P: Padding (0 here), S: Stride (1).

### 4. Training Process
- **Loss**: Categorical Crossentropy: $$ L = - \sum y_i \log(\hat{y_i}) $$
  - Measures prediction vs. true label divergence.
- **Optimizer**: Adam (adaptive learning rate) for efficient gradient descent.
- **Metrics**: Accuracy = correct predictions / total.
- **Epochs/Batch**: 10 epochs, batch=32 — balances speed/convergence.
- **Overfitting Check**: Validation loss plot; early stopping could be added.

### 5. Inference & Visualization
- **Prediction**: Softmax outputs probabilities; argmax for class.
- **Confidence**: Max probability → user-friendly % score.
- **Plot**: Training/validation curves to diagnose under/overfitting.

### 6. Machine Learning Fundamentals
- **Supervised Learning**: Labeled data (digit images + labels) trains the model.
- **Backpropagation**: Gradients flow backward to update weights.
- **Generalization**: Val split ensures model works on unseen data.
- **Why Keras/TF?**: High-level API abstracts boilerplate, focuses on concepts.

This setup achieves ~98% accuracy — comparable to basic MNIST benchmarks — while being extensible for deeper models or augmentations.

---

## 📦 Installation & Usage

### 1. Prerequisites

- Python 3.8+ (with pip)
- Microsoft Excel/CSV editor for data (optional)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Training the Model

Download [train.csv](https://media.geeksforgeeks.org/wp-content/uploads/20250407132659565012/Train.csv) from Geeks for Geeks and place in root.

Run:
```bash
python main.py
```
Outputs: Trained model (digit_recognizer_model.keras), accuracy/loss, plots.

### 4. Predicting on Custom Image

Prepare a PNG/JPG of a handwritten digit (e.g., my_digit.png in root).
Edit predict.py to set image_filename.

Run:
```bash
python predict.py
```
Outputs: Predicted digit, confidence, visualized image.


## 🎯 Credits & Extensions

- **Model & Training Code Credit:**  
  The neural network architecture and training process are based on the tutorial by GeeksforGeeks:  
  [Handwritten Digit Recognition using Neural Network](https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/)
