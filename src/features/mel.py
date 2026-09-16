import librosa
import numpy as np

from preprocessing.audio import load_audio


# --------------------------------------------------
# Mel Spectrogram Configuration
# --------------------------------------------------

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

TARGET_DURATION = 4.0
TARGET_SAMPLES = 16000 * int(TARGET_DURATION)


def extract_mel_spectrogram(filepath):
    """
    Load an audio file and convert it into a
    fixed-size log-Mel spectrogram.

    Output shape:
        (128, 126)
    """

    # --------------------------------------------------
    # 1. Load and preprocess audio
    # --------------------------------------------------

    y, sr = load_audio(filepath)

    # --------------------------------------------------
    # 2. Make audio exactly 4 seconds
    # --------------------------------------------------

    if len(y) < TARGET_SAMPLES:

        # Pad shorter recordings with zeros
        y = np.pad(
            y,
            (0, TARGET_SAMPLES - len(y)),
            mode="constant",
        )

    else:

        # Truncate longer recordings
        y = y[:TARGET_SAMPLES]

    # --------------------------------------------------
    # 3. Create Mel spectrogram
    # --------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=8000,
    )

    # --------------------------------------------------
    # 4. Convert power to decibels
    # --------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------
    # 5. Normalize spectrogram
    # --------------------------------------------------

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    return mel_db.astype(np.float32)