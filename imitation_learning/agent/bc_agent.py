import torch
from agent.networks import CNN

import torch.nn as nn
import torch.optim as optim


class BCAgent:

    def __init__(self, input_shape=(1, 96, 96), n_classes=3, learning_rate=1e-4):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)

        self.net = CNN(history_length=input_shape[0], n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification tasks
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize

        # Transform input to tensors
        X_batch = torch.tensor(X_batch, dtype=torch.float32)  # Ensure input is float for CNN
        y_batch = torch.tensor(y_batch, dtype=torch.long)  # Targets for CrossEntropyLoss must be long

        # Set model to training mode
        self.net.train()

        # Forward pass
        outputs = self.net(X_batch)
        loss = self.criterion(outputs, y_batch)

        # Backward and optimize
        self.optimizer.zero_grad()  # Clear existing gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update model parameters

        return loss.item()

        #return loss

    def predict(self, X):
        # TODO: forward pass

        # Transform input to tensor
        X = torch.tensor(X, dtype=torch.float32)

        # Set model to evaluation mode
        self.net.eval()

        # Forward pass
        with torch.no_grad():
            outputs = self.net(X)

        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
