from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = "terrain.h5"
model = tf.keras.models.load_model(MODEL_PATH)

TERRAIN_CLASSES = {
    0: "Grasslands",
    1: "Marshland",
    2: "Other",
    3: "Rocky Terrain",
    4: "Sandy Terrain",
    5: "Water"
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

IP_CAMERA_URL = "http://192.168.144.112:8080/video"  # Replace with your IP webcam URL

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html", ip_cam_url=IP_CAMERA_URL)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(255, 255))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        terrain_type = TERRAIN_CLASSES.get(predicted_class, "Unknown Terrain")

        return jsonify({
            "prediction": terrain_type,
            "file_path": file_path
        })

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/predict_live', methods=['GET'])
def predict_live():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        return jsonify({"error": "Unable to access camera"}), 400

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Could not read frame"}), 400

    frame_resized = cv2.resize(frame, (255, 255))
    frame_normalized = frame_resized.astype("float32") / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    predictions = model.predict(frame_input)
    predicted_class = np.argmax(predictions)
    terrain_type = TERRAIN_CLASSES.get(predicted_class, "Unknown Terrain")

    return jsonify({"prediction": terrain_type})

if __name__ == '__main__':
    app.run(debug=True)
