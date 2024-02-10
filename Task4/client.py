import flwr as fl

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

from functools import partial
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift

from encmodel import *
from utils import *

import argparse

# set random seed and device
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train[0 : int(len(x_train) * 0.2)]
y_train = y_train[0 : int(len(y_train) * 0.2)]  # just to get some small part of data


# Simulate federated clients (splitting the dataset)
client_data = []

for i in range(n_clients):
    start = i * len(x_train) // n_clients
    end = (i + 1) * len(x_train) // n_clients
    client_data.append((x_train[start:end], y_train[start:end]))


print(len(client_data[0][0]))


def permute_c(x):
    return np.transpose(x.astype(np.float32), (0, 3, 1, 2))


# MMD detector on each client
client_detectors = []
for x_data, _ in client_data:
    # define preprocessing function
    preprocess_fn = partial(
        preprocess_drift,
        model=encoder_model,
        device=device,
        batch_size=512,
    )

    X_ref = permute_c(x_data[0:200])
    # initialise drift detector
    detector = MMDDrift(
        X_ref,
        backend="pytorch",
        p_val=0.05,
        preprocess_fn=preprocess_fn,
        n_permutations=100,
    )
    client_detectors.append(detector)



parser = argparse.ArgumentParser(description="Flower")
parser.add_argument('--cid', type=str, help='client ID')
parser.add_argument("--mode", choices=['train', 'test'], help="client mode")
parser.add_argument('--gpu_id', help='gpu ID')
args = parser.parse_args()
print('Client:', args.cid)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# client_drift_count = {str(i): 0 for i in range(n_clients)}
client_drift_count = {args.cid: 0}

# Drift detection on client data
def handle_client_drift(x_val, cid, net):

    detector = client_detectors[int(cid)]

    detector_data = detector.predict(x_val, return_p_val=True, return_distance=True)
    is_drift = detector_data["data"].get("is_drift", None)

    p_val = detector_data["data"].get("p_val", None)
    distance = detector_data["data"].get("distance", None)

    # print("p_val:", p_val)
    # print("distance:", distance)

    isEliminated = False

    if is_drift:
        client_drift_count[cid] += 1
        print(f"Drift detected on client data {client_drift_count[cid]} times")
    else:
        print("No drift detected on client data. Continuing training.")

    if client_drift_count[cid] >= 3:
        print(f"Client {cid} has detected drift more than 3 times!")
        isEliminated = True

    return isEliminated


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_data, model, cid):
        self.client_data = client_data
        self.model = model
        self.cid = cid

    def get_parameters(self, config):
        # Return the current model parameters
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def fit(self, parameters, config):
        # Train the local model after updating it with the given parameters
        # Convert parameters from numpy arrays to torch tensors
        state_dict = {
            key: torch.from_numpy(param)
            for key, param in zip(self.model.state_dict(), parameters)
        }
        self.model.load_state_dict(state_dict)

        # Train the local model and return the new parameters, num_examples, and done flag
        new_params = [param.detach().cpu().numpy() for param in self.model.parameters()]
        num_examples = len(self.client_data["x_train"])
        return new_params, num_examples, {}

    def evaluate(self, parameters, config):
        # Perform the evaluation of the model after updating it with the given
        # parameters. Returns the loss as a float, the length of the validation
        # data, and a dict containing the accuracy
        # Convert parameters from numpy arrays to torch tensors
        state_dict = {
            key: torch.from_numpy(param)
            for key, param in zip(self.model.state_dict(), parameters)
        }
        self.model.load_state_dict(state_dict)
        # Perform evaluation
        loss, accuracy = test(
            self.model, self.client_data["x_val"], self.client_data["y_val"]
        )

        # Apply drift detection on client data
        isEliminated = handle_client_drift(self.client_data["x_val"], self.cid, self.model)
        print('isEliminated', isEliminated)

        return (
            float(loss),
            len(self.client_data["y_val"]),
            {"accuracy": float(accuracy), "isEliminated": isEliminated},
        )


def client_fn(cid: str) -> FlowerClient:
    # print("int(cid)", int(cid))
    cid = args.cid
    # print('args.cid', args.cid)
    x_data, y_data = client_data[int(cid)]
    # x_data = np.array(x_data)
    x_data = permute_c(x_data)
    # y_data = np.array(y_data)

    x_train = x_data[0 : int(len(x_data) * 0.8)]
    y_train = y_data[0 : int(len(y_data) * 0.8)]

    x_val = x_data[int(len(x_data) * 0.8) :]
    y_val = y_data[int(len(y_data) * 0.8) :]

    all_data = []
    all_data.extend((x_train, y_train, x_val, y_val, int(cid)))

    model = Encoder(encoding_dim).to(device)

    return FlowerClient(
        client_data={
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
        },
        model=model,
        cid=cid,
    ).to_client()


fl.client.start_client(
    server_address="[::]:8080",
    client_fn=client_fn,
)
