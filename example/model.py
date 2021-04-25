import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets

torch.manual_seed(0)

class Network(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
    self.conv2 = nn.Conv2d(6, 6, kernel_size=5)
    self.conv3 = nn.Conv2d(6, 1, kernel_size=5)

    self.linear1 = nn.Linear(256, 20)
    self.linear2 = nn.Linear(20, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)

    x = x.view(-1, 256)
    # print(x.shape)

    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.softmax(x)

    return x

class MnistModel:
  def __init__(self):
    self.network = Network()
    self.optimizer = optim.Adam(self.network.parameters())

  def predict(self, x):
    return self.network(x)

  def train(self, epochs=1):
    # Get data:

    print("Beginning training...")

    mnist_trainset = datasets.MNIST(root='/Users/maciej/Gaze/example/data', train=True, download=True,
      transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
    ]))
    batch_size_train = 64

    self.train_loader = torch.utils.data.DataLoader(
      mnist_trainset,
      batch_size=batch_size_train, shuffle=True)

    self.network.train()  
    for i in range(epochs):
      print(f"Epoch #{i}...")
      for batch_idx, (data, target) in enumerate(self.train_loader):
        self.optimizer.zero_grad()
        output = self.network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()

  def test(self):

    mnist_testset = datasets.MNIST(root='/Users/maciej/Gaze/example/data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ]))

    batch_size_test = 1000
    self.test_loader = torch.utils.data.DataLoader(
      mnist_testset,
      batch_size=batch_size_test, shuffle=True)

    self.network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in self.test_loader:
        output = self.network(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(self.test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

  def saveModel(self):
    torch.save(self.network.state_dict(), "mnist_model")  
    torch.save(self.optimizer.state_dict(), "mnist_opt_state")
  def loadModel(self, state, opt_state):
    self.network.load_state_dict(torch.load(state))
    self.optimizer.load_state_dict(torch.load(opt_state))
