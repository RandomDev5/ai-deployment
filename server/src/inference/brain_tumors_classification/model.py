import torch.nn as nn


# This block will be explained quickly through the part 1 of the deployment mechanism
class BrainTumorClassifier(nn.Module):
    """
    A convolutional neural network model for brain tumor classification.

    This model consists of:
    - Two convolutional layers for feature extraction.
    - A fully connected classifier for predicting tumor types.

    Attributes:
        features (nn.Sequential): Convolutional and pooling layers.
        classifier (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, num_classes):

        super(BrainTumorClassifier, self).__init__()
        """
        Initialize the BrainTumorClassifier.

        Args:
            num_classes (int): Number of output classes.
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
