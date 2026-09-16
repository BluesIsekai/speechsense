import matplotlib.pyplot as plt
import librosa
import librosa.display

from dataset import load_dataset


if __name__ == "__main__":
    df = load_dataset()

    # Take the first sample
    sample = df.iloc[0]

    y, sr = librosa.load(sample["filepath"], sr=None, mono=True)

    print("File:", sample["filename"])
    print("Emotion:", sample["emotion"])
    print("Sample rate:", sr)
    print("Duration:", len(y) / sr)

    # Waveform
    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(y, sr=sr)

    plt.title(f"Waveform - {sample['emotion']}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000,
    )

    mel_db = librosa.power_to_db(mel, ref=y.max())

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {sample['emotion']}")
    plt.tight_layout()
    plt.show()