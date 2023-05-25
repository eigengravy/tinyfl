import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
trainset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
trainloader = DataLoader(trainset, num_workers=4, batch_size=batch_size)

testset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
testloader = DataLoader(testset, batch_size=batch_size)

# Nueral netwrok model for the image classification
# flatten: flattens the input tensor
# layers: equential layers of the model
class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    # Foward pass of model
    # Args: x: Input tensor
    # Returns Output tensor
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Creates instance of the model
# Returns intsance model
def create_model() -> Model:
    model = Model().to(device)
    return model

# Trains the model, Args: model: model being trained, epochs: number of training epochs
def train_model(model, epochs: int):
    global trainloader
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(epochs):
        model.train()
        for _, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = nn.CrossEntropyLoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Tests the model, Args: Model
# Returns accuracy and loss
def test_model(model):
    size = len(testloader.dataset)
    batches = len(testloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += nn.CrossEntropyLoss()(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= batches
    correct /= size
    return 100 * correct, test_loss

# Averages all the weights from multiple clients
# Returns the averages weights
def fedavg_models(weights):
    avg = copy.deepcopy(weights[0])
    for i in range(1, len(weights)):
        for key in avg:
            avg[key] += weights[i][key]
        avg[key] = torch.div(avg[key], len(weights))
    return avg
