import torch
from torch import nn
from train import MyCNN
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

model = MyCNN()
loss = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters())

trainedstate = torch.load("trainedmodel.pth", weights_only=False)
model.load_state_dict(trainedstate['model_state_dict()'])
optimiser.load_state_dict(trainedstate['optimiser_state_dict'])

model.eval()

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

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

test(test_dataloader, model, loss)

