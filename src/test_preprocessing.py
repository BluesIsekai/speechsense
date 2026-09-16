from dataset import load_dataset
from preprocessing.audio import load_audio


if __name__ == "__main__":
    df = load_dataset()

    sample = df.iloc[0]

    y, sr = load_audio(sample["filepath"])

    print("File:", sample["filename"])
    print("Emotion:", sample["emotion"])
    print("Sample rate:", sr)
    print("Samples:", len(y))
    print("Duration:", len(y) / sr)
    print("Min amplitude:", y.min())
    print("Max amplitude:", y.max())