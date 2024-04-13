import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from alibi_detect.cd import MMDDrift

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Load data from CSV
df = pd.read_csv("file.csv")  # my df has drift at time steps 9500, 20000, 30500.
df = df[
    0:20000
]  # I took small subset, now I have 2 different concepts and 1 drift at time step 9500
xd1_data = df[9500:10500]
df = df.drop(df.index[9500:10500])

X = df[["X1", "X2"]].values
y = df["class"].values
Xd1 = xd1_data[["X1", "X2"]].values
yd1 = xd1_data["class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_ref = X_train[:8500]  # I took part where concept is stable (Before drift)


class Encoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Flatten(),
        )

        linear_output_size = self._get_linear_output_size((1, input_dim))

        self.linear = nn.Linear(linear_output_size, encoding_dim)

    def _get_linear_output_size(self, shape):
        batch_size = 1
        input = torch.rand(batch_size, *shape)
        output_feat = self.encoder(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.encoder(x)
        return self.linear(x)


# Initialize the model
model = Encoder(input_dim=2, encoding_dim=32).to(device).eval()


def train(model, x_train, y_train, num_epochs=5):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    # Convert numpy array to PyTorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}: train loss {epoch_loss}")

    return model


def test(model, x_test, y_test):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    print("total", total)
    accuracy = correct / total
    return loss, accuracy


for epoch in range(5):
    train(model, X_train, y_train, num_epochs=5)
    # model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=True)
    loss, accuracy = test(model, X_test, y_test)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

loss, accuracy = test(model, X_test, y_test)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")


mmd = MMDDrift(x_ref=X_ref, backend="pytorch", p_val=0.05)
# Detect drift
X_test_tensor = torch.tensor(Xd1)
X_test_numpy = X_test_tensor.numpy()
preds = mmd.predict(X_test_numpy, return_p_val=True, return_distance=True)

is_drift = preds["data"]["is_drift"]
p_val = preds["data"]["p_val"]
print("Drift? {}".format("Yes!" if preds["data"]["is_drift"] else "No!"))
print(f'p-value: {preds["data"]["p_val"]:.3f}')
