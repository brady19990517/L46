import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from flwr_experimental.baseline.dataset.dataset import create_partitioned_dataset

class PartitionedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], int(self.Y[idx]))


def load_data():

    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainset, testset, num_examples

def load_local_partitioned_data(client_id, iid_fraction: float, num_partitions: int):
    """Creates a dataset for each worker, which is a partition of a larger dataset."""
    
    # Each worker loads the entire dataset, and then selects its partition
    # determined by its `client_id` (happens internally below)
    trainset, testset, num_examples = load_data()
    
    train_loader = DataLoader(trainset, batch_size=len(trainset))
    test_loader = DataLoader(testset, batch_size=len(testset))

    (x_train, y_train), (x_test, y_test) = next(iter(train_loader)), next(iter(test_loader))
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()

    (train_partitions, test_partitions), _ = create_partitioned_dataset(
        ((x_train, y_train), (x_test, y_test)), iid_fraction, num_partitions)
 
    x_train, y_train = train_partitions[client_id]
    torch_partition_trainset = PartitionedDataset(torch.Tensor(x_train), y_train)
    x_test, y_test = test_partitions[client_id]
    torch_partition_testset = PartitionedDataset(torch.Tensor(x_test), y_test )
    return torch_partition_trainset, torch_partition_testset