import numpy as np
from preprocessing import extract_features
from sklearn.preprocessing import StandardScaler


def predict_custom_audio(model, scaler, le, audio_path, target_sr, n_mfcc, n_chroma, n_sc):
    # Extract features from audio file
    features = extract_features(audio_path, target_sr, n_mfcc, n_chroma, n_sc)

    # Standardize the features
    features = scaler.transform([features])

    # Predict the label using the ANN model
    prediction = model.predict(features)
    prediction_index = np.argmax(prediction)
    label = le.inverse_transform([prediction_index])[0]

    return label
