import numpy as np
import pandas as pd

from dataset import load_dataset
from features.mfcc import extract_mfcc


OUTPUT_PATH = "data/processed/mfcc.csv"


if __name__ == "__main__":
    df = load_dataset()

    features = []
    labels = []
    actors = []

    print(f"Extracting MFCCs from {len(df)} files...")

    for i, row in df.iterrows():
        feature_vector = extract_mfcc(row["filepath"])

        features.append(feature_vector)
        labels.append(row["emotion"])
        actors.append(row["actor"])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    X = np.array(features)

    feature_columns = [
        f"mfcc_{i + 1}"
        for i in range(X.shape[1])
    ]

    feature_df = pd.DataFrame(
        X,
        columns=feature_columns,
    )

    feature_df["emotion"] = labels
    feature_df["actor"] = actors

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone!")
    print("Feature matrix:", X.shape)
    print("Saved to:", OUTPUT_PATH)