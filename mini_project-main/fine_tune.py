import os
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =========================================
# 1. Load your existing trained model
# =========================================
model_path = "trained_data_model.h5"
model = tf.keras.models.load_model(model_path)
print("✅ Loaded model for fine-tuning.")

# =========================================
# 2. Unfreeze top layers for fine-tuning
# =========================================
print(f"Total layers in model: {len(model.layers)}")
for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# 3. Data preparation with augmentation
# =========================================
IMG_SIZE = (192, 192)
BATCH_SIZE = 32

TRAIN_DIR = "C:\\Indian-Currency-Recognition-System-master\\data\\train"
VAL_DIR = "C:\\Indian-Currency-Recognition-System-master\\data\\val"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================================
# 4. Callbacks for training
# =========================================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
checkpoint = callbacks.ModelCheckpoint('model_finetuned_B8.h5', save_best_only=True, monitor='val_accuracy')

steps_per_epoch = math.ceil(train_ds.samples / BATCH_SIZE)
val_steps = math.ceil(val_ds.samples / BATCH_SIZE)

# =========================================
# 5. Train (fine-tune)
# =========================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# =========================================
# 6. Save the fine-tuned model
# =========================================
model.save("model_final_B8_finetuned.h5")
print("✅ Fine-tuned model saved as model_final_B8_finetuned.h5")

# =========================================
# 7. Evaluate model with confusion matrix
# =========================================
print("\n📊 Evaluating model...")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

# Predictions
Y_pred = model.predict(val_ds)
y_pred = np.argmax(Y_pred, axis=1)

classes = ["₹10", "₹20", "₹50", "₹100", "₹200", "₹500"]
print("\n🧾 Classification Report:\n")
print(classification_report(val_ds.classes, y_pred, target_names=classes))

print("\n📊 Confusion Matrix:\n")
cm = confusion_matrix(val_ds.classes, y_pred)
print(cm)

# =========================================
# 8. Plot confusion matrix heatmap
# =========================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix — Fine-Tuned Model")
plt.show()
