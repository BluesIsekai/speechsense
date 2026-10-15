
import os
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.mel_dataset import MelSpectrogramDataset
from models.cnn_depthwise import SpeechEmotionCNN


# ==================================================
# Configuration
# ==================================================

FEATURES_PATH = "data/processed/mel_spectrograms.npy"
LABELS_PATH = "data/processed/mel_labels.csv"

# Separate checkpoint: preserve the previous best model.
MODEL_PATH = (
    "data/processed/"
    "best_cnn_depthwise_l2_1e3_earlystop.pth"
)

HISTORY_PATH = (
    "data/processed/"
    "history_cnn_depthwise_l2_1e3_earlystop.csv"
)

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
NUM_CLASSES = 8

PATIENCE = 7
MIN_DELTA = 1e-4


# ==================================================
# Device
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# Evaluation helper
# ==================================================

def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    total = 0
    predictions_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(
                DEVICE, non_blocking=True
            )
            labels = labels.to(
                DEVICE, non_blocking=True
            )

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            predictions_all.extend(
                predictions.cpu().numpy()
            )
            labels_all.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = np.mean(
        np.asarray(predictions_all)
        == np.asarray(labels_all)
    )

    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        predictions_all,
        labels_all,
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpeechSense Depthwise CNN Training")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(1, 17),
        augment=False,
    )

    val_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(17, 21),
    )

    test_dataset = MelSpectrogramDataset(
        FEATURES_PATH,
        LABELS_PATH,
        actors=range(21, 25),
    )

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

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
    # Model and optimizer
    # --------------------------------------------------

    model = SpeechEmotionCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"\nWeight decay: {WEIGHT_DECAY}")
    print(f"Early stopping patience: {PATIENCE}")
    print("Checkpoint selection: validation loss")

    os.makedirs("data/processed", exist_ok=True)

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(
                DEVICE, non_blocking=True
            )
            labels = labels.to(
                DEVICE, non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            train_correct += (
                (predictions == labels).sum().item()
            )
            train_total += labels.size(0)

        train_loss = (
            train_running_loss / train_total
        )
        train_accuracy = (
            train_correct / train_total
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_f1,
            _,
            _,
        ) = evaluate(model, val_loader, criterion)

        improved = (
            val_f1 > best_val_f1 + MIN_DELTA
        )

        if improved:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            status = " <- BEST VAL LOSS"
        else:
            epochs_without_improvement += 1
            status = ""

        # Record metrics for every epoch.
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
            f"{status}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping triggered at "
                f"epoch {epoch + 1}."
            )
            print(
                f"Validation loss did not improve "
                f"for {PATIENCE} consecutive epochs."
            )
            break

    # --------------------------------------------------
    # Save history
    # --------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    # --------------------------------------------------
    # Load best validation-loss checkpoint
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Loading Best Model")
    print("=" * 60)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Validation accuracy at selected epoch: "
          f"{best_val_accuracy:.4f}")
    print(f"Validation macro F1 at selected epoch: "
          f"{best_val_f1:.4f}")

    # --------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------

    print("\nEvaluating selected model on test set...")

    (
        test_loss,
        test_accuracy,
        test_f1,
        test_predictions,
        test_labels,
    ) = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    print(f"Selected epoch: {best_epoch}")
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    print(f"Test macro F1:  {test_f1:.4f}")

    print(f"\nCheckpoint saved to:\n{MODEL_PATH}")
    print(f"Training history saved to:\n{HISTORY_PATH}")

    # --------------------------------------------------
    # Plot loss
    # --------------------------------------------------

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation Loss",
    )
    plt.axvline(
        best_epoch,
        linestyle="--",
        label="Selected Epoch",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Depthwise CNN: Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "data/processed/loss_earlystop.png",
        dpi=150,
    )
    plt.show()

    # --------------------------------------------------
    # Plot accuracy
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="Validation Accuracy",
    )
    plt.axvline(
        best_epoch,
        linestyle="--",
        label="Selected Epoch",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Depthwise CNN: Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "data/processed/accuracy_earlystop.png",
        dpi=150,
    )
    plt.show()
