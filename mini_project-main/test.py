import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = tf.keras.models.load_model("trained_data_model.h5")

# Test dataset
IMG_SIZE = (192, 192)
BATCH_SIZE = 32

test_gen = ImageDataGenerator(rescale=1./255)
test_ds = test_gen.flow_from_directory(
    "C:\\Indian-Currency-Recognition-System-master\\data\\test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Predict
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_ds.classes

# Class names
class_names = list(test_ds.class_indices.keys())

# Metrics
print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred_classes))

# Calculate number of steps for evaluation
steps = test_ds.samples // test_ds.batch_size

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_ds, steps=steps)
print(f"\nTest Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
