import torch

from data.mel_dataset import (
    MelSpectrogramDataset,
    ID_TO_EMOTION,
)


FEATURES_PATH = (
    "data/processed/mel_spectrograms.npy"
)

LABELS_PATH = (
    "data/processed/mel_labels.csv"
)


if __name__ == "__main__":

    # --------------------------------------------------
    # Create training dataset
    # --------------------------------------------------

    train_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(1, 17),
    )

    # --------------------------------------------------
    # Create validation dataset
    # --------------------------------------------------

    val_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(17, 21),
    )

    # --------------------------------------------------
    # Create test dataset
    # --------------------------------------------------

    test_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(21, 25),
    )

    # --------------------------------------------------
    # Print dataset sizes
    # --------------------------------------------------

    print("Dataset sizes")
    print("-------------")

    print(
        f"Training:   {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print(
        f"Test:       {len(test_dataset)}"
    )

    # --------------------------------------------------
    # Inspect one sample
    # --------------------------------------------------

    spectrogram, label = train_dataset[0]

    print("\nSample")
    print("------")

    print(
        "Spectrogram shape:",
        spectrogram.shape,
    )

    print(
        "Spectrogram dtype:",
        spectrogram.dtype,
    )

    print(
        "Label ID:",
        label.item(),
    )

    print(
        "Emotion:",
        ID_TO_EMOTION[label.item()],
    )

    print(
        "Min:",
        spectrogram.min().item(),
    )

    print(
        "Max:",
        spectrogram.max().item(),
    )

    # --------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    batch_x, batch_y = next(iter(loader))

    print("\nDataLoader")
    print("----------")

    print(
        "Batch input shape:",
        batch_x.shape,
    )

    print(
        "Batch labels shape:",
        batch_y.shape,
    )