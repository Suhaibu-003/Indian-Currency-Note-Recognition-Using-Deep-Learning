from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import playsound  # ✅ added import

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = load_model("trained_data_model.h5")

# ⚠️ Must match training order:
# {'10': 0, '100': 1, '20': 2, '200': 3, '50': 4, '500': 5}
classes = ["10", "100", "20", "200", "50", "500"]

# --------------------------------------------------
# Flask app setup
# --------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction_text="⚠️ No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction_text="⚠️ No file selected")

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((192, 192))  # match your training input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 192, 192, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]
    confidence = float(np.max(prediction) * 100)

    # Map audio for notes
    audio_map = {
        "10": r"C:\\Users\SANJAY\Desktop\\Indian-Currency-Recognition-System-master\\audio\\10.mp3",
        "20": r"C:\\Users\SANJAY\Desktop\\Indian-Currency-Recognition-System-master\\audio\\20.mp3",
        "50": r"C:\\Users\SANJAY\Desktop\\Indian-Currency-Recognition-System-master\\audio\\50.mp3",
        "100": r"C:\\Users\SANJAY\Desktop\\Indian-Currency-Recognition-System-master\\audio\\100.mp3",
        "200": r"C:\\Users\SANJAY\Desktop\\Indian-Currency-Recognition-System-master\\audio\\200.mp3",
        "500": r"C:\\Users\SANJAY\Desktop\\Indian-Currency-Recognition-System-master\\audio\\500.mp3",
    }

    # ✅ Play audio if available
    if predicted_label in audio_map:
        try:
            playsound.playsound(audio_map[predicted_label], True)
        except Exception as e:
            print(f"⚠️ Audio playback error: {e}")

    # ✅ Return result to webpage
    return render_template(
        "index.html",
        prediction_text=f"💵 Predicted Note: ₹{predicted_label} ({confidence:.2f}% confidence)",
        uploaded_image=filepath,
    )


# --------------------------------------------------
# Run the app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
