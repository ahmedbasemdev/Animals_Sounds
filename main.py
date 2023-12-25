from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_custom_audio
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello from Flask!'


# Load the ANN model
model = load_model('files/ann_model.h5')


# Load the label encoder and scaler
scaler = joblib.load("files/scaler.joblib")
le = joblib.load("files/le.joblib")


# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    uploaded_file = request.files['file']

    # Save the uploaded file to disk
    file_path = os.path.join(
        app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)

    # Map the predicted class index to the class label
    predicted_class = predict_custom_audio(
        model=model, scaler=scaler, le=le, audio_path=file_path, target_sr=22050, n_mfcc=40, n_chroma=12, n_sc=7)

    # Return the predicted class as a JSON response
    response = {'predicted_class': predicted_class}
    print(response)
    return jsonify(response)


# Define the main function
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    app.run(host="0.0.0.0", port=5000)
