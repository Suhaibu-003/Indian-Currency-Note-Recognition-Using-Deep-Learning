# validate.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --------------------------
# Load trained model
# --------------------------
MODEL_PATH = "trained_data_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --------------------------
# Dataset paths & parameters
# --------------------------
VAL_DIR = "C:\\Indian-Currency-Recognition-System-master\\data\\val"
BATCH_SIZE = 32

# --------------------------
# Determine model input size dynamically
# --------------------------
input_shape = model.input_shape  # (None, height, width, channels)
IMG_SIZE = (input_shape[1], input_shape[2])
print(f"ℹ Using input image size: {IMG_SIZE}")

# --------------------------
# Prepare validation data generator
# --------------------------
val_gen = ImageDataGenerator(rescale=1./255)
val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --------------------------
# Calculate steps
# --------------------------
steps = val_ds.samples // BATCH_SIZE
if val_ds.samples % BATCH_SIZE != 0:
    steps += 1

# --------------------------
# Evaluate the model
# --------------------------
val_loss, val_acc = model.evaluate(val_ds, steps=steps)
print(f"✅ Validation Accuracy: {val_acc:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")

# --------------------------
# Optional: per-class accuracy
# --------------------------
try:
    import numpy as np
    from sklearn.metrics import classification_report

    val_preds = model.predict(val_ds, steps=steps)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = val_ds.classes
    class_labels = list(val_ds.class_indices.keys())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
except ImportError:
    print("ℹ sklearn not installed, skipping per-class metrics.")
