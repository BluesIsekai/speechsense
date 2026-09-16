from pathlib import Path

import librosa
import pandas as pd

from dataset import load_dataset


def inspect_audio(df):
    sample_rates = []
    durations = []
    channels = []

    for filepath in df["filepath"]:
        y, sr = librosa.load(filepath, sr=None, mono=False)

        sample_rates.append(sr)

        if y.ndim == 1:
            channels.append(1)
            duration = len(y) / sr
        else:
            channels.append(y.shape[0])
            duration = y.shape[-1] / sr

        durations.append(duration)

    df["sample_rate"] = sample_rates
    df["duration"] = durations
    df["channels"] = channels

    return df


if __name__ == "__main__":
    df = load_dataset()

    print(f"Total files: {len(df)}")
    print("\nInspecting audio...")

    df = inspect_audio(df)

    print("\nSample rate distribution:")
    print(df["sample_rate"].value_counts())

    print("\nChannel distribution:")
    print(df["channels"].value_counts())

    print("\nDuration statistics:")
    print(df["duration"].describe())

    print("\nDuration by emotion:")
    print(
        df.groupby("emotion")["duration"]
        .agg(["count", "mean", "min", "max"])
        .round(3)
    )