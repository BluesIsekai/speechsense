import torch
import torch.nn as nn


class SpeechEmotionCNN(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        # --------------------------------------------------
        # Convolutional feature extractor
        # --------------------------------------------------

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # --------------------------------------------------
        # Global average pooling
        # --------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# --------------------------------------------------
# Test model
# --------------------------------------------------

if __name__ == "__main__":

    model = SpeechEmotionCNN()

    print(model)

    # Fake batch matching our actual data
    x = torch.randn(
        32,
        1,
        128,
        126,
    )

    output = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nNumber of parameters:")

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"{parameters:,}")