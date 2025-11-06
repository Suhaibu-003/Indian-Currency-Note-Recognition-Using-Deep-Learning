import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN numerical differences
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU only

from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------
# 🧭 Paths (Update these if needed)
# -------------------------------------------------
base_dir = "C:\\Indian-Currency-Recognition-System-master\\data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# -------------------------------------------------
# 🧼 Convert grayscale images to RGB
# -------------------------------------------------
def convert_images_to_rgb(base_dir):
    print(f"\n🔄 Checking and converting grayscale images to RGB in: {base_dir}\n")
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        img.save(path)
                        count += 1
                except Exception as e:
                    print(f"⚠️ Could not process: {path} | Error: {e}")
    if count > 0:
        print(f"✅ Converted {count} images to RGB.\n")
    else:
        print("✅ All images already in RGB.\n")

convert_images_to_rgb(train_dir)
convert_images_to_rgb(val_dir)
convert_images_to_rgb(test_dir)

# -------------------------------------------------
# 🖼️ Image and training settings
# -------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# -------------------------------------------------
# 🧩 Data Generators
# -------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False  # Important for evaluation
)

# -------------------------------------------------
# 🧠 Load EfficientNetB0 (Pretrained on ImageNet)
# -------------------------------------------------
print("\n🧠 Loading EfficientNetB0 with ImageNet weights...\n")

try:
    base_model = EfficientNetB0(
        weights="C:\\Indian-Currency-Recognition-System-master\\efficientnetb0_notop.h5",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    print("✅ Loaded ImageNet weights successfully.\n")
except Exception as e:
    print(f"⚠️ Could not load ImageNet weights. Using untrained model instead.\nError: {e}")
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------------------------
# 🔧 Custom Classification Layers
# -------------------------------------------------
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------------------------
# ⚙️ Compile Model
# -------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------
# 🏋️ Train Model
# -------------------------------------------------
print("\n🚀 Starting training EfficientNetB0 on CPU...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------------------------------
# 🧪 Evaluate on Test Set
# -------------------------------------------------
print("\n📊 Evaluating model on test dataset...\n")
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# -------------------------------------------------
# 💾 Save Trained Model
# -------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("eff_trained_model.h5")

print("\n✅ Training and testing completed successfully!")
print("📁 Model saved as: models/eff_trained_model.h5")
