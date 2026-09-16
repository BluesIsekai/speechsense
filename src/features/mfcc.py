import librosa
import numpy as np

from preprocessing.audio import load_audio


N_MFCC = 13


def extract_mfcc(filepath):
    y, sr = load_audio(filepath)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
    )

    # Summarize the time dimension
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
    ])

    return features