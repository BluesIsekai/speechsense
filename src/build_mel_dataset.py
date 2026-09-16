import os

import numpy as np
import pandas as pd

from dataset import load_dataset
from features.mel import extract_mel_spectrogram


OUTPUT_FEATURES = "data/processed/mel_spectrograms.npy"
OUTPUT_LABELS = "data/processed/mel_labels.csv"


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_dataset()

    print(f"Total recordings: {len(df)}")

    # --------------------------------------------------
    # 2. Prepare output directory
    # --------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    # --------------------------------------------------
    # 3. Extract Mel spectrograms
    # --------------------------------------------------

    spectrograms = []
    labels = []
    actors = []

    print("\nExtracting Mel spectrograms...")

    for i, row in df.iterrows():

        mel = extract_mel_spectrogram(
            row["filepath"]
        )

        spectrograms.append(mel)
        labels.append(row["emotion"])
        actors.append(row["actor"])

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(df)}"
            )

    # --------------------------------------------------
    # 4. Convert to NumPy array
    # --------------------------------------------------

    X = np.array(
        spectrograms,
        dtype=np.float32,
    )

    print("\nExtraction complete.")

    print("Feature shape:", X.shape)
    print("Data type:", X.dtype)

    # --------------------------------------------------
    # 5. Save spectrograms
    # --------------------------------------------------

    np.save(
        OUTPUT_FEATURES,
        X,
    )

    # --------------------------------------------------
    # 6. Save labels and actors
    # --------------------------------------------------

    metadata = pd.DataFrame(
        {
            "emotion": labels,
            "actor": actors,
        }
    )

    metadata.to_csv(
        OUTPUT_LABELS,
        index=False,
    )

    # --------------------------------------------------
    # 7. Final information
    # --------------------------------------------------

    print("\nSaved:")
    print(
        f"Spectrograms: {OUTPUT_FEATURES}"
    )
    print(
        f"Labels:       {OUTPUT_LABELS}"
    )

    print("\nFinal dataset:")
    print(f"Samples: {X.shape[0]}")
    print(f"Mel bands: {X.shape[1]}")
    print(f"Time frames: {X.shape[2]}")