import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageChops

# Configuration
MODEL_PATH = 'digit_recognizer_model.keras'
IMG_SIZE = 28
TARGET = 20  # scale digit to fit ~20px like MNIST

def load_and_prep_image(image_path):
    img = Image.open(image_path).convert("L")

    # Invert if background is light (more robust than only top-left is even better, but ok)
    if img.getpixel((0, 0)) > 128:
        img = ImageOps.invert(img)

    arr = np.array(img)
    mask = arr > 30  # “ink” threshold in inverted image (white-ish strokes)

    if mask.any():
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        img = img.crop((x0, y0, x1, y1))

    # Resize to TARGET while keeping aspect ratio
    w, h = img.size
    if w > h:
        new_w = TARGET
        new_h = max(1, int(round(h * TARGET / w)))
    else:
        new_h = TARGET
        new_w = max(1, int(round(w * TARGET / h)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste centered into 28x28
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    canvas.paste(img, ((IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2))

    # Center by intensity “center of mass”
    a = np.array(canvas).astype(np.float32)
    s = a.sum()
    if s > 0:
        y_idx = np.arange(IMG_SIZE)[:, None]
        x_idx = np.arange(IMG_SIZE)[None, :]
        cy = (a * y_idx).sum() / s
        cx = (a * x_idx).sum() / s
        shift_x = int(round(IMG_SIZE / 2 - cx))
        shift_y = int(round(IMG_SIZE / 2 - cy))
        canvas = ImageChops.offset(canvas, shift_x, shift_y)

    img_array = (np.array(canvas) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img_array, canvas


def main():
    # --- CHANGE THIS TO YOUR IMAGE FILENAME ---
    image_filename = 'my_digit.png' 
    # ------------------------------------------

    try:
        # 1. Load the saved model
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # 2. Preprocess the image
        print(f"Processing image: {image_filename}...")
        processed_img, original_img = load_and_prep_image(image_filename)

        # 3. Make prediction
        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # 4. Show results
        print(f"\n-----------------------------")
        print(f"PREDICTION: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"-----------------------------")

        # Visual check
        plt.figure(figsize=(4, 4))
        plt.imshow(processed_img.reshape(28, 28), cmap='gray')
        plt.title(f"AI sees: {predicted_digit} ({confidence:.1f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()