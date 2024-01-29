import torch

n_clients = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1, "num_cpus": 1}


print(client_resources)
