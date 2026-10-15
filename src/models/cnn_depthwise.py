
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise convolution followed by pointwise convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()

        padding = kernel_size // 2

        # Spatial filtering independently per input channel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.depthwise_bn = nn.BatchNorm2d(in_channels)

        # Mix information across channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.pointwise_bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)

        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.activation(x)

        return x


class SpeechEmotionCNN(nn.Module):
    """Depthwise-separable CNN for SpeechSense."""

    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(
                1, 32, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 1: 32 -> 64
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),

            # Block 2: 64 -> 128
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),

            # Block 3: 128 -> 256
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SpeechEmotionCNN()

    x = torch.randn(32, 1, 128, 126)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Parameters: {parameters:,}")
