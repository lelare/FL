import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from alibi_detect.cd import MMDDrift
import torch.nn as nn

# Load data from CSV
df = pd.read_csv("file.csv")
df = df[0:20000]
xd1_data = df[9500:10500]
df = df.drop(df.index[9500:10500])

# Split features and labels
X = df[["X1", "X2"]].values
y = df["class"].values
Xd1 = xd1_data[["X1", "X2"]].values
yd1 = xd1_data["class"].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define reference distribution
X_ref = X_train[:8500]

# Initialize the MMD detector
mmd = MMDDrift(x_ref=X_ref, backend="pytorch", p_val=0.05)


# Train function
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test_model(model, X_test, y_test):
    # Convert data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

    accuracy = accuracy_score(y_test, predicted.numpy())
    return accuracy


# Define model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Train the model
model = SimpleModel(input_size=2, hidden_size=64, output_size=2)
train_model(model, X_train, y_train)

# Test the model
accuracy = test_model(model, X_test, y_test)
print("Accuracy:", accuracy)

# Detect drift
X_test_tensor = torch.tensor(Xd1)
X_test_numpy = X_test_tensor.numpy()
preds = mmd.predict(X_test_numpy, return_p_val=True, return_distance=True)

# Access the results
is_drift = preds["data"]["is_drift"]
p_val = preds["data"]["p_val"]
# threshold = preds["data"]["threshold"]
# distance = preds["data"]["distance"]
# distance_threshold = preds["data"]["distance_threshold"]
print("Drift? {}".format("Yes!" if preds["data"]["is_drift"] else "No!"))
print(f'p-value: {preds["data"]["p_val"]:.3f}')
