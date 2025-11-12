import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
### CONSTANTS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
NUM_EPOCHS = 5

class NeuralNet(nn.Module):
    def __init__(self):
      super(NeuralNet, self).__init__()
      self.fc1 = nn.Linear(in_features=28*28, out_features=128)
      self.fc2 = nn.Linear(in_features=128, out_features=128)
      self.fc3 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
      # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
      # because nn.Linear expects 1D feature vectors
      x = torch.flatten(x, start_dim=1)

      # Pass data through 1st fully connected layer
      x = self.fc1(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 2nd fully connected layer
      x = self.fc2(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 3rd fully connected layer
      x = self.fc3(x)

      # Before passing x to the (log) softmax function,
      # the values in x are called *logits*.

      # Apply softmax to x (in log domain)
      log_probs = F.log_softmax(x, dim=1)

      return log_probs
    
def define_data():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data',
                                train=True,
                                download=True,
                                transform=transform)

    test_dataset = datasets.MNIST('./data',
                                train=False,
                                transform=transform)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64,
    shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
    )
    return train_loader, test_loader

train_loader, test_loader = define_data()
model = NeuralNet().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.NLLLoss()
loss_fn_test = nn.NLLLoss(reduction='sum')


def train_eval_model():
    losses = []
    for epoch in range(NUM_EPOCHS):
        print("-"*20, f"Epoch {epoch}", "-"*20)
        model.train() # modus flag
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            log_probs = model(data)
            loss = loss_fn(log_probs, target)
            loss.backward() # berechnet gradienten via backpropagation
            optimizer.step() # weights werden geupdated
            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(f"Train Epoch {epoch} | Loss: {loss.item()}")
        print(f"\nAverage train loss in epoch {epoch}: {np.mean(losses[-len(train_loader):])}")

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
        # deaktivieren von gradienten tracking
        # keine backpropagation ergebnisse gespeichert
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                log_probs = model(data)
                test_loss += loss_fn_test(log_probs, target).item()
                # Ermitteln von höchster Log-Wahrscheinlichkeit aus den 10 Tensors
                pred = torch.argmax(log_probs, dim=1)
                correct += (pred == target).sum().item()
        test_loss /= len(test_loader.dataset)
        avg_correct = correct / len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{len(test_loader.dataset)} \
                ({100. * avg_correct:.0f}%)\n")
    return losses
def plot_loss_curve(losses):
    # average every 10th batch loss
    print(len(losses))
    losses_smoothed = np.array(losses).reshape(-1, 10).mean(axis=1)
    steps = np.arange(len(losses))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b', alpha=0.5)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.axhline(y=-np.log(0.1), color='r', linestyle='--', label='Baseline Loss')
    plt.legend(loc='right')
    plt.savefig('training_loss_mnist.png', dpi=300)


losses = train_eval_model()
plot_loss_curve(losses=losses)
