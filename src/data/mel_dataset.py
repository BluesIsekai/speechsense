import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


# --------------------------------------------------
# Emotion labels
# --------------------------------------------------

EMOTION_TO_ID = {
    "angry": 0,
    "calm": 1,
    "disgust": 2,
    "fearful": 3,
    "happy": 4,
    "neutral": 5,
    "sad": 6,
    "surprised": 7,
}


ID_TO_EMOTION = {
    value: key
    for key, value in EMOTION_TO_ID.items()
}


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class MelSpectrogramDataset(Dataset):

    def __init__(
        self,
        features_path,
        labels_path,
        actors,
        augment=False,
    ):

        # Load spectrograms
        self.features = np.load(
            features_path
        )

        # Load metadata
        metadata = pd.read_csv(
            labels_path
        )

        # Select requested actors
        mask = metadata["actor"].isin(
            actors
        )

        self.features = self.features[mask]

        self.labels = [
            EMOTION_TO_ID[emotion]
            for emotion in metadata.loc[
                mask, "emotion"
            ]
        ]

        self.labels = np.array(
            self.labels,
            dtype=np.int64,
        )

        # Whether to apply augmentation
        self.augment = augment

    # --------------------------------------------------
    # Dataset length
    # --------------------------------------------------

    def __len__(self):
        return len(self.features)

    # --------------------------------------------------
    # SpecAugment
    # --------------------------------------------------

    def apply_spec_augment(
        self,
        spectrogram,
    ):
        """
        Apply frequency masking and time masking.

        Input shape:
            (128, 126)
        """

        # Convert to tensor
        spectrogram = torch.tensor(
            spectrogram,
            dtype=torch.float32,
        )

        # ----------------------------------------------
        # Frequency masking
        # ----------------------------------------------

        frequency_mask_size = np.random.randint(
            0,
            17,
        )

        if frequency_mask_size > 0:

            frequency_start = np.random.randint(
                0,
                128 - frequency_mask_size + 1,
            )

            spectrogram[
                frequency_start:
                frequency_start + frequency_mask_size,
                :
            ] = 0

        # ----------------------------------------------
        # Time masking
        # ----------------------------------------------

        time_mask_size = np.random.randint(
            0,
            21,
        )

        if time_mask_size > 0:

            time_start = np.random.randint(
                0,
                126 - time_mask_size + 1,
            )

            spectrogram[
                :,
                time_start:
                time_start + time_mask_size,
            ] = 0

        return spectrogram

    # --------------------------------------------------
    # Get sample
    # --------------------------------------------------

    def __getitem__(self, index):

        spectrogram = self.features[index]

        # Apply augmentation only when enabled
        if self.augment:

            spectrogram = self.apply_spec_augment(
                spectrogram
            )

        else:

            spectrogram = torch.tensor(
                spectrogram,
                dtype=torch.float32,
            )

        # Add channel dimension
        # (128, 126) -> (1, 128, 126)
        spectrogram = spectrogram.unsqueeze(0)

        # Label
        label = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return spectrogram, label