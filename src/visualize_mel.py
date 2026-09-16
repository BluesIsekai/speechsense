import matplotlib.pyplot as plt
import librosa.display

from dataset import load_dataset
from features.mel import extract_mel_spectrogram


if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Pick one sample
    sample = df.iloc[0]

    # Extract Mel spectrogram
    mel = extract_mel_spectrogram(
        sample["filepath"]
    )

    # Display
    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        mel,
        x_axis="time",
        y_axis="mel",
        sr=16000,
        hop_length=512,
        fmax=8000,
    )

    plt.colorbar(
        format="%+2.0f"
    )

    plt.title(
        f"Log-Mel Spectrogram - {sample['emotion']}"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency")

    plt.tight_layout()

    plt.show()