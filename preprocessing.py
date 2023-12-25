import librosa
from librosa import util
import numpy as np


def extract_features(audio_path, target_sr, n_mfcc, n_chroma, n_sc):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    length = 3

    # Pad or truncate the audio signal to desired length
    if len(audio) < length * sr:
        audio = util.pad_center(audio, size=length * sr)
    else:
        audio = audio[:length * sr]

    # Extract MFCC coefficients
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Extract Chroma Frequencies
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    # Extract Root Mean Square Energy
    rms = librosa.feature.rms(y=audio)

    # Extract Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

    # Extract Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    chroma_mean = np.mean(chroma, axis=1)
    chroma_var = np.var(chroma, axis=1)

    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_var = np.var(spectral_contrast, axis=1)

    rms_mean = np.mean(rms, axis=1)
    rms_var = np.var(rms, axis=1)

    spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
    spectral_centroid_var = np.var(spectral_centroid, axis=1)

    zero_crossing_rate_mean = np.mean(zero_crossing_rate, axis=1)
    zero_crossing_rate_var = np.var(zero_crossing_rate, axis=1)

    # Combine all features
    features = np.concatenate([mfcc_mean, mfcc_var, chroma_mean, chroma_var, spectral_contrast_mean, spectral_contrast_var,
                              rms_mean, rms_var, spectral_centroid_mean, spectral_centroid_var, zero_crossing_rate_mean, zero_crossing_rate_var])

    return features
