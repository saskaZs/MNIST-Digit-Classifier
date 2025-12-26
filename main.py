import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Configuration
CSV_PATH = 'train.csv'
MODEL_SAVE_PATH = 'digit_recognizer_model.keras'
IMG_SIZE = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2  # 20% of data used for validation
RANDOM_SEED = 42

def load_and_preprocess_data(csv_path):
    """
    Loads data from CSV, separates labels and pixels,
    and reshapes/normalizes the images.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: '{csv_path}' not found. Please place it in the project root.")

    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)

    # Separate labels (Y) and pixels (X)
    # Assuming the first column is 'label'
    y = data.iloc[:, 0].values
    x = data.iloc[:, 1:].values

    # Normalize pixel values (0-255 -> 0-1)
    x = x / 255.0

    # Reshape into images: (num_samples, 28, 28, 1)
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print(f"Data loaded. Shape: {x.shape}")
    return x, y

def build_model():
    """
    Builds a Sequential Neural Network.
    """
    model = models.Sequential([
        # Flatten the 28x28x1 images into a 1D vector
        layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # Hidden layer 1
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Dropout to prevent overfitting
        
        # Hidden layer 2
        layers.Dense(64, activation='relu'),
        
        # Output layer (10 digits)
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    """
    Plots the training and validation accuracy/loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

def main():
    try:
        # 1. Load Data
        X, y = load_and_preprocess_data(CSV_PATH)

        # 2. Split into Training and Validation sets
        print(f"Splitting data (Test size: {TEST_SIZE})...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # 3. Build Model
        print("Building model...")
        model = build_model()
        model.summary()

        # 4. Train Model
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val)
        )

        # 5. Evaluate
        print("\nFinal Evaluation on Validation Set:")
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")

        # 6. Save Model
        model.save(MODEL_SAVE_PATH)
        print(f"\nModel saved successfully to '{MODEL_SAVE_PATH}'")

        # 7. Visualize Training Results
        plot_history(history)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()