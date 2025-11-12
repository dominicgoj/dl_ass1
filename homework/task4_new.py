import torch
from models import BinaryNeuralNet
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, training_data, test_data, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    test_loss_fn = nn.NLLLoss()

    ### NLLLoss, use softmax, log probability

    training_epoch_losses = []
    testing_epoch_losses = []
    accuracies = []
    for epoch in range(epochs):
        print('-'*20, f'Epoch {epoch}', '-'*20)
        model.train()
        training_batch_losses = []
        
        for batch_idx, (data, target) in enumerate(training_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logp = model(data)
            loss = loss_fn(logp, target)
            loss.backward()
            optimizer.step()
            training_batch_losses.append(loss.item())
        
        avg_epoch_loss = sum(training_batch_losses)/len(training_data)
        training_epoch_losses.append(avg_epoch_loss)
        
        print(f'\nAverage train loss in epoch {epoch}: {avg_epoch_loss}')
            
        
        model.eval()
        test_batch_losses = []
        correct = 0
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                logp = model(data)
                loss = test_loss_fn(logp, target).item()
                test_batch_losses.append(loss)
                pred = logp.argmax(dim=1)
                correct += (pred == target).sum().item()
            avg_test_batch_loss = sum(test_batch_losses)/len(test_data)

        avg_correct = correct / len(test_data.dataset)
        testing_epoch_losses.append(avg_test_batch_loss)
        accuracies.append(correct/len(test_data.dataset))
        print(f'Test set: Average loss: {avg_test_batch_loss:.4f}, Accuracy: {correct}/{len(test_data.dataset)} ({100. * avg_correct:.0f}%)\n')
        
    return training_epoch_losses, testing_epoch_losses, accuracies

        #print(f'\nAverage train loss in epoch {epoch}: {np.mean(losses[-len(training_data):])}')



SEED = 302
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

np.random.seed(302)

g = torch.Generator()
g.manual_seed(302)

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=302)
y_train[y_train < 2], y_test[y_test < 2] = 0, 0
y_train[y_train >= 2], y_test[y_test >= 2] = 1, 1

scaler = StandardScaler()
transformed_X_train_split = scaler.fit_transform(X=X_train)
transformed_X_test_split = scaler.transform(X=X_test)

X_train_tensor, y_train_tensor = torch.from_numpy(transformed_X_train_split).float(), torch.from_numpy(y_train).long()
X_test_tensor, y_test_tensor = torch.from_numpy(transformed_X_test_split).float(), torch.from_numpy(y_test).long()

train_dataset = torch.utils.data.TensorDataset(
    X_train_tensor, y_train_tensor
)

test_dataset = torch.utils.data.TensorDataset(
    X_test_tensor, y_test_tensor
)

training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, generator=g)

model = BinaryNeuralNet()
model.to(device)
lr = 0.001
epochs = 15
train_losses, test_losses, accuracies = train_model(
    training_data=training_dataloader,
    test_data=test_dataloader,
    model=model,
    epochs=epochs,
    learning_rate=lr
)