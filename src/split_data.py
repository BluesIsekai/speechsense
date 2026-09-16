import pandas as pd


INPUT_PATH = "data/processed/mfcc.csv"


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    train_actors = list(range(1, 17))
    val_actors = list(range(17, 21))
    test_actors = list(range(21, 25))

    train = df[df["actor"].isin(train_actors)]
    val = df[df["actor"].isin(val_actors)]
    test = df[df["actor"].isin(test_actors)]

    print("Dataset split:")
    print(f"Training:   {len(train)} samples")
    print(f"Validation: {len(val)} samples")
    print(f"Test:       {len(test)} samples")

    print("\nTraining actors:")
    print(sorted(train["actor"].unique()))

    print("\nValidation actors:")
    print(sorted(val["actor"].unique()))

    print("\nTest actors:")
    print(sorted(test["actor"].unique()))

    print("\nEmotion distribution:")
    print("\nTraining:")
    print(train["emotion"].value_counts().sort_index())

    print("\nValidation:")
    print(val["emotion"].value_counts().sort_index())

    print("\nTest:")
    print(test["emotion"].value_counts().sort_index())