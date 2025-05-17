import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class MyCNN(nn.Module):
  def __init__(self):
    super().__init__()

    # output size = (input size - kernel size + 2padding) / stride + 1
    # output conv1: 32-3+2(1)/ 1 + 1= 28+2+1 = 32x32x24
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)
    # output pool: (31-3+2(0))/2 + 1 = 16x16x24
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    # output conv2: (16-5+2(1))/1 + 1 = 14x14x48
    self.conv2 = nn.Conv2d(in_channels=24, out_channels=48,kernel_size=5, stride=1, padding=1)
    # output pool2: (14-2+2(0))/2 + 1 = 7x7x48
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    # output conv3: (7-5+2(1))/1 + 1 = 5x5x96
    self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5, stride=1, padding=1)
    # output pool3: (5-2+2(0))/2 + 1 = 2x2x96
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    # output conv4: (2-2+2(1))/1 + 1 = 3x3x128
    self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=2, stride=1, padding=1)
    # output pool4: (3-2+2(0))/2 + 1 = 1x1x128
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    self.fc1 = nn.Linear(in_features=128*1*1, out_features=512)
    self.fc2 = nn.Linear(in_features=512, out_features=256)
    self.fc3 = nn.Linear(in_features=256, out_features=128)
    self.fc4 = nn.Linear(in_features=128, out_features=10)

    self.dropout = nn.Dropout(p=0.3)

    self.bn1 = nn.BatchNorm2d(24)
    self.bn2 = nn.BatchNorm2d(48)
    self.bn3 = nn.BatchNorm2d(96)
    self.bn4 = nn.BatchNorm2d(128)

  def forward(self, x):
    x = self.bn1(self.conv1(x))
    x = self.pool(F.relu(x))
    x = self.bn2(self.conv2(x))
    x = self.pool2(F.relu(x))
    x = self.bn3(self.conv3(x))
    x = self.pool3(F.relu(x))
    x = self.bn4(self.conv4(x))
    x = self.pool4(F.relu(x))
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

def train(dataloader, model, loss_fn, optimiser, device="cpu"):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    optimiser.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimiser.step()

    if batch % (len(dataloader) // 6) == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    
    

def test(dataloader, model, loss_fn, device="cpu"):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")
  return test_loss


if __name__ == "__main__":
  model = MyCNN().to(device="cpu")
  loss = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3, cooldown=2, threshold=0.05)
  
  train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.3),
     transforms.RandomVerticalFlip(p=0.3),
     transforms.RandomCrop(32, padding=3),
     transforms.RandomRotation(degrees=20),
     transforms.ColorJitter(0.3,0.3,0.3,0.1),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform
  )

  test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
  )

  batch_size = 16
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
  
  epochs = 40
  
  for t in range(epochs):
      print(f"Epoch {t+1} \n-------------------------------")
      train(train_dataloader, model, loss, optimiser)
      validation_loss = test(test_dataloader, model, loss)
      scheduler.step(validation_loss)
      print(f" Learning rate: {optimiser.param_groups[0]['lr']} \n")
  torch.save({'model_state_dict': model.state_dict(),
              'optimiser_state_dict': optimiser.state_dict()}, "trainedmodel.pth")
  print("Done :) Saved trained model to trainedmodel.pth")