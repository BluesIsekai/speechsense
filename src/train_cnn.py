import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

NUM_CLASSES = 8


# ==================================================
# Device
# ==================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpeechSense CNN Training")
    print("=" * 60)

    # --------------------------------------------------
    # Device information
    # --------------------------------------------------

    print("\nDevice:")
    print(DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    print("\nLoading datasets...")

    # Training actors: 1-16
    # No augmentation for this experiment
    train_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(1, 17),
        augment=False,
    )

    # Validation actors: 17-20
    val_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(17, 21),
    )

    # Test actors: 21-24
    test_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(21, 25),
    )

    print(
        f"Training samples:   {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Test samples:       {len(test_dataset)}"
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = SpeechEmotionCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    print("\nModel:")
    print(model)

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        # ==============================================
        # Training
        # ==============================================

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            # Move data to GPU
            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(
                outputs,
                labels,
            )

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            # ------------------------------------------
            # Training statistics
            # ------------------------------------------

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # ==============================================
        # Validation
        # ==============================================

        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(
                    DEVICE,
                    non_blocking=True,
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True,
                )

                # Forward pass
                outputs = model(inputs)

                # Validation loss
                loss = criterion(
                    outputs,
                    labels,
                )

                val_running_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                # Predictions
                predictions = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

                val_total += labels.size(0)

        val_loss = (
            val_running_loss / val_total
        )

        val_accuracy = (
            val_correct / val_total
        )

        # --------------------------------------------------
        # Store history
        # --------------------------------------------------

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            os.makedirs(
                "data/processed",
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            saved = " ← BEST"

        else:

            saved = ""

        # --------------------------------------------------
        # Print epoch results
        # --------------------------------------------------

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
            f"{saved}"
        )

    # ==================================================
    # Load best model
    # ==================================================

    print("\n")
    print("=" * 60)
    print("Loading Best Model")
    print("=" * 60)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    # ==================================================
    # Test evaluation
    # ==================================================

    model.eval()

    test_correct = 0
    test_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            # Forward pass
            outputs = model(inputs)

            # Predictions
            predictions = outputs.argmax(
                dim=1
            )

            test_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            test_total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    test_accuracy = (
        test_correct / test_total
    )

    # ==================================================
    # Final results
    # ==================================================

    print(
        f"\nBest Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"\nBest model saved to:"
        f"\n{MODEL_PATH}"
    )

    # ==================================================
    # Plot loss
    # ==================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_losses,
        label="Training Loss",
    )

    plt.plot(
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "CNN Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ==================================================
    # Plot accuracy
    # ==================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_accuracies,
        label="Training Accuracy",
    )

    plt.plot(
        val_accuracies,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "CNN Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()