from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os
import tensorflow as tf

app = Flask(__name__)

# ==========================================
# 1. Load model safely and verify GPU usage
# ==========================================
MODEL_PATH = "model_final_B8_finetuned.h5"
model = load_model(MODEL_PATH)

print("\n✅ Model loaded successfully from:", MODEL_PATH)

# ==========================================
# 2. Define your currency note classes
# ==========================================
classes = ["10", "100", "20", "200", "50", "500"]

# ==========================================
# 3. Routes
# ==========================================
@app.route("/")
def index():
    return render_template("indexv2.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        img_data = data["image"]
        img_bytes = base64.b64decode(img_data.split(",")[1])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess (resize, normalize)
        img = img.resize((192, 192))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = classes[predicted_index]
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "label": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500

# ==========================================
# 4. Run the Flask app
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)
