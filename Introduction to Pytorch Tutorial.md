# Introduction to Pytorch tutorial

Pytorch like most machine learning frameworks includes:

- Working with data
- Creating model
- Optimizing model parameters (Training)
- Loading & Saving trained models
- Inference

 This tutorial introduces you to a complete ML workflow implemented in PyTorch, with links to learn more about each of these concepts. We are going to use FashionMNIST classification as the example.

## Working with Data

Pytorch has two different data objects:

1. ```python
   from torch.utils.data import DataLoader
   ```

2. ```python
   from torch.utils.data import Dataset
   ```

PyTorch offers domain-specific libraries such as [TorchText](https://pytorch.org/text/stable/index.html), [TorchVision](https://pytorch.org/vision/stable/index.html), and [TorchAudio](https://pytorch.org/audio/stable/index.html), all of which include datasets. For this tutorial, we will be using a TorchVision dataset. 

```python
from torchvision import datasets
from torchtext import datasets
from torchaudio import datasets
```

Every TorchVision `Dataset` includes two arguments: `transform` and `target_transform` to modify the samples and labels respectively.

Download data from datasets module in torchvision:

```python
training_data = datasets.FashionMNIST(
  root='data'
  train=True, #False for test dataset
  download=True,
  transform=Totensor())
```

We pass the Dataset as an argument to DataLoader to that wraps an iterable over the dataset, and supports automatic batching, sampling, shuffling, and multiprocess data loading. 

```python
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

for X, y in test_dataloader:
  print("Shape of X [N, C, H, W]: ", X.shape)
  print("Shape of y: ", y.shape, y.dtype)
  break
```

Out:

```python
Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
Shape of y:  torch.Size([64]) torch.int64
```



## Creating Models

To create a neural net in Pytorch you need to create a class that inherits the nn.Module object. You then define layers in the init function and create a function for the forward pass through the network. You can also put the model on GPU is you have one to use:

```python
import torch
from torch import nn

class JacobsPytorchNN(nn.Module):
  def __init__(self):
    super(JacobsPytorchNN, self).__init__()
    self.flatten = nn.Flatten()
    self.network_architecture = nn.Sequential(
      nn.Linear(28*28,512),
      nn.ReLU(),
      nn.Linear(512,512),
      nn.ReLU(),
      nn.Linear(512,10),
      nn.ReLU()
    )
	
  def forward(self, x):
    x = self.flatten(x)
    logits = self.network_architecture(x)
    return logits

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
  
model = JacobsPytorchNN().to(device)
```



## Optimizing the Model Parameter

To train we need a loss function and optimizer algo:

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

First we need to define a training loop & a evaluation function to get eval metric on test set:

```python
def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X,y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    
    # Compute Loss
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
      
def evaluation(dataloader, model, loss_fn, optimizer):
  size = len(dataloader)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0,0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

Next we write our training loop using the functions above:

```python
num_epochs = 5
for i in range(epochs):
  print(f"Epoch {i+1}\n------------------------"")
  train(train_dataloader, model, loss_fn, optimizer)
  test(test_dataloader, model, loss_fn, optimizer)
print("Done!")
```



## Loading & Saving Models

Loading models:

```python
model = JacobsPytorchNN()
model.load_state_dict(torch.load("model.pth"))
```

Saving models:

```python
torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")
```



## Inference

Making predictions on one new sample:

```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

