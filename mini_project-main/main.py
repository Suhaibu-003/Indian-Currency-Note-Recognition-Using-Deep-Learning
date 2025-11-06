import os
import glob
import time
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import playsound

# -----------------------------
# Open file dialog to select image
# -----------------------------
print("<<<<<<<< Enter the File You Want to Open >>>>>>>\n")
#time.sleep(2)

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

print("<<<<< You have chosen file path as :  ")
print(file_path)
time.sleep(2)

# -----------------------------
# Load training images (for evaluation)
# -----------------------------
PATH = os.getcwd()
data_path = PATH + '/data/train'
major = os.listdir(data_path)
full_path = []
image_labels = []
all_labels = ["10","20","50","100","200","500"]
num_classes = len(all_labels)

print("Loading file structure...\n")
for a in major:
    full_path.append(os.path.join(data_path, a))

print("Loading training images...\n")
train_images = []
for folder in full_path:
    label = os.path.basename(folder)
    images_in_folder = 0
    for file in glob.glob(os.path.join(folder, "*.jpg")):
        img = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)
        img = cv.resize(img, (192, 192))
        train_images.append(img)
        image_labels.append(all_labels.index(label))
        images_in_folder += 1
    print(f"The total number of images in {label} = {images_in_folder}")

print(f"The total number of images in data = {len(train_images)}")
time.sleep(2)

# -----------------------------
# Preprocess
# -----------------------------
X_train = np.array(train_images, np.float32) / 255.
image_labels = to_categorical(image_labels, num_classes=num_classes)

mean_img = X_train.mean(axis=0)
std_dev = X_train.std(axis=0)
X_norm = (X_train - mean_img) / std_dev

X_norm, image_labels = shuffle(X_norm, image_labels, random_state=0)
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X_norm, image_labels, test_size=0.2, random_state=7)

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("C:\\Indian-Currency-Recognition-System-master\\trained_data_model.h5")

print("<<<<<< This is the Model We Trained >>>>>>>> \n")
time.sleep(2)
model.summary()

score = model.evaluate(Xvalid, Yvalid, verbose=0)
print(f"\nOur model's accuracy: {score[1]*100:.2f}%\n")

# -----------------------------
# Compile (optional if already compiled)
# -----------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----------------------------
# Predict the selected image
# -----------------------------
print("<<<<<<<<< Predicting Input Image >>>>>>>>\n")
time.sleep(2)

img = cv.cvtColor(cv.imread(file_path), cv.COLOR_BGR2RGB)
img = cv.resize(img, (192, 192))
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.

# Use model.predict and argmax (predict_classes is deprecated)
pred_probs = model.predict(img)
pred_class = np.argmax(pred_probs, axis=1)[0]

print("Your predicted class index:", pred_class)
detected_label = all_labels[pred_class]
print(f"\nDetected denomination: Rs. {detected_label}\n")

# -----------------------------
# Play audio based on denomination
# -----------------------------
audio_map = {
    "10": 'C:\\Users\\SANJAY\\Desktop\\Indian-Currency-Recognition-System-master\\audio\\10.mp3',
    "20": 'C:\\Users\\SANJAY\\Desktop\\Indian-Currency-Recognition-System-master\\audio\\20.mp3',
    "50": 'C:\\Users\\SANJAY\\Desktop\\Indian-Currency-Recognition-System-master\\audio\\50.mp3',
    "100": 'C:\\Users\\SANJAY\\Desktop\\Indian-Currency-Recognition-System-master\\audio\\100.mp3',
    "500": 'C:\\Users\\SANJAY\\Desktop\\Indian-Currency-Recognition-System-master\\audio\\500.mp3',
    # Add paths for other notes if needed
}

if detected_label in audio_map:
    playsound.playsound(audio_map[detected_label], True)

time.sleep(2)
print("<<<<<< Thank You for using our Currency Recognition System >>>>> \n")
time.sleep(2)
