import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network

        # Input channels based on history length, defaulting to 1 if history_length is 0 (assuming grayscale images)
        in_channels = history_length if history_length > 0 else 1

        # Define the layers of a convolutional neural network
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Assuming input image size is 96x96, calculate the size after convolutions and pooling
        # self.final_conv_size = self._get_conv_output_size([96, 96])

        # Fully connected layers
        self.fc1 = nn.Linear(36864, 64)
        self.fc2 = nn.Linear(64, 64)


    # def _get_conv_output_size(self, size):
    #     # Helper function to calculate the output size after all convolutions and pooling
    #     size = [((size[0] - 4) // 2), ((size[1] - 4) // 2)]  # after conv1 and pool
    #     size = [((size[0] - 4) // 2), ((size[1] - 4) // 2)]  # after conv2 and pool
    #     size = [((size[0] - 2) // 1), ((size[1] - 2) // 1)]  # after conv3
    #     return size


    def forward(self, x):
        # TODO: compute forward pass

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
