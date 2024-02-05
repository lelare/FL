import flwr as fl

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import *


class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 512, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of the output from the convolutional layers
        conv_output_size = self._get_conv_output_size((1, 28, 28))

        self.linear = nn.Linear(conv_output_size, encoding_dim)

    def _get_conv_output_size(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.encoder(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.encoder(x)
        return self.linear(x)


# Define the encoding dimension
encoding_dim = 32

# Instantiate the Encoder model
encoder_model = Encoder(encoding_dim).to(device).eval()


BATCH_SIZE = 32


def train(model, x_train, y_train, num_epochs=5):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    train_dataset = TensorDataset(
        torch.tensor(x_train), torch.tensor(y_train.argmax(axis=1))
    )
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    return model


def test(model, x_test, y_test):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(x_test), torch.tensor(y_test.argmax(axis=1))
    )
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    print("total", total)
    accuracy = correct / total
    return loss, accuracy
