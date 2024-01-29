import flwr as fl

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

from functools import partial
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift


import random

# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import TensorDataset, DataLoader

from utils import *


class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 512, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, encoding_dim),
        )

    def forward(self, x):
        return self.encoder(x)


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

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
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
