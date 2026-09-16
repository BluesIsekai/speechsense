from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

MODALITIES = {
    1: "full-AV",
    2: "video-only",
    3: "audio-only-speech",
    4: "audio-only-song",
}


def parse_ravdess_file(filepath: Path):
    parts = filepath.stem.split("-")

    modality_code = int(parts[0])
    emotion_code = int(parts[2])
    actor = int(parts[6])

    return {
        "filepath": str(filepath),
        "filename": filepath.name,
        "actor": actor,
        "modality_code": modality_code,
        "modality": MODALITIES.get(modality_code, "unknown"),
        "emotion_code": emotion_code,
        "emotion": EMOTIONS[emotion_code],
    }


def load_dataset():
    records = []

    for filepath in DATA_DIR.rglob("*.wav"):
        records.append(parse_ravdess_file(filepath))

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_dataset()

    print(f"Total recordings: {len(df)}")
    print()

    print("Modality distribution:")
    print(df["modality"].value_counts())
    print()

    print("Emotion distribution:")
    print(df["emotion"].value_counts().sort_index())
    print()

    print("Speech-only recordings:")
    speech = df[df["modality"] == "audio-only-speech"]
    print(len(speech))
    print()

    print("Speech emotion distribution:")
    print(speech["emotion"].value_counts().sort_index())
    print()

    print("First 5 samples:")
    print(df.head())