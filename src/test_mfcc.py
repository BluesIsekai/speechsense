import numpy as np

from dataset import load_dataset
from features.mfcc import extract_mfcc


if __name__ == "__main__":
    df = load_dataset()

    sample = df.iloc[0]

    features = extract_mfcc(sample["filepath"])

    print("File:", sample["filename"])
    print("Emotion:", sample["emotion"])
    print("Feature shape:", features.shape)
    print("Features:")
    print(np.round(features, 3))