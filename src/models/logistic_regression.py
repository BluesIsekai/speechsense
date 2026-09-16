import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/processed/mfcc.csv"


def split_data(df):
    """
    Split the dataset by actor.

    Actors 1-16  -> Training
    Actors 17-20 -> Validation
    Actors 21-24 -> Test
    """

    train = df[df["actor"].between(1, 16)]
    val = df[df["actor"].between(17, 20)]
    test = df[df["actor"].between(21, 24)]

    # Select only MFCC features
    feature_columns = [
        column
        for column in df.columns
        if column.startswith("mfcc_")
    ]

    X_train = train[feature_columns]
    y_train = train["emotion"]

    X_val = val[feature_columns]
    y_val = val["emotion"]

    X_test = test[feature_columns]
    y_test = test["emotion"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Total samples: {len(df)}")

    # --------------------------------------------------
    # 2. Split dataset
    # --------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_data(df)

    print("\nDataset Split")
    print("-------------")
    print(f"Training:   {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test:       {len(X_test)} samples")

    # --------------------------------------------------
    # 3. Standardize features
    # --------------------------------------------------

    print("\nStandardizing MFCC features...")

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit only on training data.
    X_train_scaled = scaler.fit_transform(X_train)

    # Validation and test use the same scaler.
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # 4. Create Logistic Regression model
    # --------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    # --------------------------------------------------
    # 5. Train
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train_scaled,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------
    # 6. Validation evaluation
    # --------------------------------------------------

    val_predictions = model.predict(X_val_scaled)

    print("\n")
    print("=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    print(f"Accuracy: {val_accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 7. Test evaluation
    # --------------------------------------------------

    test_predictions = model.predict(X_test_scaled)

    print("\n")
    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    print(f"Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 8. Confusion Matrix
    # --------------------------------------------------

    print("\nGenerating confusion matrix...")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        xticks_rotation=45,
    )

    plt.title(
        "Logistic Regression - Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------

    print("\n")
    print("=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")