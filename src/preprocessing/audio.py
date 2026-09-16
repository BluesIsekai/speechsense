import librosa
import numpy as np


TARGET_SR = 16000


def load_audio(filepath):
    y, sr = librosa.load(
        filepath,
        sr=TARGET_SR,
        mono=True,
    )

    # Normalize amplitude
    y = librosa.util.normalize(y)

    return y, TARGET_SR