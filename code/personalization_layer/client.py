from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import grpc
import argparse

import cifar


import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

class BaseNet(nn.Module):
    def __init__(self) -> None:
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class PersonalNet(nn.Module):
    def __init__(self) -> None:
        super(PersonalNet, self).__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FullModel(nn.Module):
    def __init__(self, base, personal) -> None:
        super(FullModel, self).__init__()
        self.base = base
        self.personal = personal
    def forward(self, x):
        x = self.base(x)
        x = self.personal(x)
        return x


basenet = BaseNet().to(DEVICE)
personalnet = PersonalNet().to(DEVICE)
fullnet = FullModel(basenet, personalnet)

# trainloader, testloader, num_examples = cifar.load_data()

class CifarClient(fl.client.NumPyClient):

    def __init__(self, cid, base_model, personal_net, fullnet, trainset, testset):
        self.cid = cid
        self.base_model = base_model
        self.personal_net = personal_net
        self.fullnet = fullnet
        self.trainset = trainset
        self.testset = testset

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.base_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.base_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.basenet.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.set_parameters(parameters)
        train(self.fullnet, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader), {}

    def evaluate(self, parameters, config):

        testloader = DataLoader(self.testset, batch_size=32, shuffle=False)

        self.set_parameters(parameters)
        loss, accuracy = test(self.fullnet, testloader)
        return float(loss), len(testloader), {"accuracy": float(accuracy)}

def load_model():
    return fullnet

def start_client(client_id, num_partitions, iid_fraction=1.0, 
                 server_address="localhost:8080", log_host=None):
    # Configure logger
    fl.common.logger.configure(f"client_{client_id}", host=log_host)

    print(f"Loading data for client {client_id}")
    trainset, testset = cifar.load_local_partitioned_data(
        client_id=client_id, 
        iid_fraction=iid_fraction, 
        num_partitions=num_partitions)
    

    # Start client
    print(f"Starting client {client_id}")
    client = CifarClient(client_id, basenet, personalnet, fullnet, trainset, testset)
    # f'{exp_name}_iid-fraction_{iid_fraction}')

    print(f"Connecting to {server_address}")

    try:
        # There's no graceful shutdown when gRPC server terminates, so we try/except
        fl.client.start_numpy_client(server_address, client)
    except grpc._channel._MultiThreadedRendezvous:
        print(f"Client {client_id}: shutdown")



if __name__ == "__main__":
    num_clients = 10
    num_partitions = 10
    iid_fraction = 0.5

    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--cid", type=int, required=True, help="Client CID (no default)")

    args, _ = parser.parse_known_args()
    
    # fl.client.start_numpy_client("localhost:8080", client=CifarClient(100,basenet,personalnet,fullnet,0,0))
    start_client(client_id=args.cid, num_partitions=num_partitions, iid_fraction=iid_fraction, server_address="localhost:8080")
    print(f"Started {num_clients} clients")
