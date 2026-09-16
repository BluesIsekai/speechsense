import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

from data.mel_dataset import MelSpectrogramDataset
from models.cnn import SpeechEmotionCNN


# ==================================================
# Configuration
# ==================================================

FEATURES_PATH = (
    "data/processed/mel_spectrograms.npy"
)

LABELS_PATH = (
    "data/processed/mel_labels.csv"
)

MODEL_PATH = (
    "data/processed/best_cnn.pth"
)

BATCH_SIZE = 32

NUM_CLASSES = 8


# ==================================================
# Device
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpeechSense CNN Evaluation")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(21, 25),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = SpeechEmotionCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # --------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------

    print("\nLoading best CNN checkpoint...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    print("Checkpoint loaded.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(inputs)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print("\n")
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "angry",
                "calm",
                "disgust",
                "fearful",
                "happy",
                "neutral",
                "sad",
                "surprised",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    print("Generating confusion matrix...")

    ConfusionMatrixDisplay.from_predictions(
        all_labels,
        all_predictions,
        display_labels=[
            "angry",
            "calm",
            "disgust",
            "fearful",
            "happy",
            "neutral",
            "sad",
            "surprised",
        ],
        xticks_rotation=45,
    )

    plt.title(
        "CNN - Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()