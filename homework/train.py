import torch
import torch.nn as nn
import numpy as np
def train_eval_model(train_loader,
                     val_loader,
                     model,
                     epochs:int=5,
                     learning_rate:float=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses = []

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    avg_train_epoch_loss = []
    avg_val_epoch_loss = []

    for epoch in range(epochs):
        #print("-"*10, f"Epoch {epoch}", "-"*10)
        epoch_train_loss = 0
        model.train() # modus flag for training
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_losses.append(loss.item())
            
        avg_train_epoch_loss.append(epoch_train_loss/len(train_loader))
        
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = loss_fn(pred, target)
                epoch_val_loss += loss.item()

        avg_val_epoch_loss.append(epoch_val_loss/len(val_loader))
        
        #print(f"Epoch {epoch}: Train Loss: {epoch_train_loss/len(train_loader):.4f}, Val Loss: {epoch_val_loss/len(val_loader):.4f}\n")

    return avg_train_epoch_loss, avg_val_epoch_loss

def train_eval_model_binary(train_loader,
                     test_loader,
                     model,
                     epochs:int=5,
                     learning_rate:float=0.001,
                     lossfunc='bceloss'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if lossfunc == 'bceloss':
        loss_fn = nn.BCELoss()
    elif lossfunc == 'nllloss':
        loss_fn = nn.NLLLoss()

    avg_train_epoch_loss = []
    avg_val_epoch_loss = []
    avg_accuracies = []
    correct = 0
    for epoch in range(epochs):
        epoch_train_loss = 0
        model.train() # modus flag for training
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            if lossfunc == 'bceloss':
                data, target = data.to(device), target.to(device)   
            else:
                data, target = data.to(device), target.to(device).view(-1).long()
          
            prob = model(data)  
            loss = loss_fn(prob, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_epoch_loss.append(epoch_train_loss/len(train_loader))
        model.eval()
        epoch_val_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                
                if lossfunc == 'bceloss':
                    data, target = data.to(device), target.to(device)
                else:
                    data, target = data.to(device), target.to(device).view(-1).long()
                prob = model(data)
                loss = loss_fn(prob, target)
                epoch_val_loss += loss.item()
                if lossfunc == 'bceloss':
                    pred_cls = (prob >= 0.5).float()
                elif lossfunc == 'nllloss':
                    pred_cls = prob.argmax(dim=1)
                correct += (pred_cls == target).sum().item()
        
        avg_correct = correct/len(test_loader.dataset)
        avg_accuracies.append(avg_correct)
        

        avg_val_epoch_loss.append(epoch_val_loss/len(test_loader))
        
        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss/len(train_loader):.4f}, \
               Val Loss: {epoch_val_loss/len(test_loader):.4f} \
                Accuracy: {round(avg_correct*100, 1)}%\n")

    return avg_train_epoch_loss, avg_val_epoch_loss, avg_accuracies