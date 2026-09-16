import numpy as np

from dataset import load_dataset
from features.mel import extract_mel_spectrogram


if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Select one sample
    sample = df.iloc[0]

    # Extract Mel spectrogram
    mel = extract_mel_spectrogram(
        sample["filepath"]
    )

    print("File:", sample["filename"])
    print("Emotion:", sample["emotion"])

    print("\nMel Spectrogram")
    print("----------------")
    print("Shape:", mel.shape)
    print("Data type:", mel.dtype)
    print("Minimum:", mel.min())
    print("Maximum:", mel.max())
    print("Mean:", mel.mean())
    print("Standard deviation:", mel.std())