import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

SEED = 302
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)


g = torch.Generator()
g.manual_seed(SEED)

feature_names = ['MedInc',
                'HouseAge',
                'AveRooms',
                'AveBedrms', 
                'Population',
                'AveOccup',
                'Latitude',
                'Longitude']

TASK2_MODEL_ARCHITECTURES = [
        [8, 16, 8, 1],
        [8, 16, 32, 16, 8, 1],
        [8, 16, 32, 64, 32, 16, 8, 1],
        [8, 16, 32, 64, 128, 64, 32, 16, 8, 1],
        [8, 16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 1],
        [8, 16, 16, 8, 1],
        [8, 16, 32, 32, 16, 8, 1],
        [8, 16, 32, 64, 64, 32, 16, 8, 1],
        [8, 16, 32, 64, 128, 128, 64, 32, 16, 8, 1],
        [8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8, 1],
    ]

TASK2_HYPERPARAMETERS = {
        'learning_rate': [
            0.01,
            0.005,
        ],
        'epochs': [
            5, 10, 20, 50
        ]
    }


TASK3_FINAL_MODEL_ARCHITECTURE = [8, 16, 32, 16, 8, 1]
TASK3_LR = 0.01
TASK3_EPOCHS = 25

TASK4_LEARNING_RATE = 0.05
TASK4_EPOCHS = 15

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
        
        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss/len(train_loader):.3f}, \
               Val Loss: {epoch_val_loss/len(test_loader):.3f} \
                Accuracy: {round(avg_correct*100, 1)}%\n")

    return avg_train_epoch_loss, avg_val_epoch_loss, avg_accuracies


class NeuralNet(nn.Module):
    """
    Model structure for task 2 (testing different model architectures)
    """
    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList([
            nn.Linear(in_features=self.layer_sizes[i], out_features=self.layer_sizes[i+1]) for i in range(len(layer_sizes) -1)
        ])
        
    def forward(self, x):
        # no flatting necessary because dataset is already flat
        # in constrast to images which have px and channel
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class BinaryNeuralNet(nn.Module):
    """
    Variant 1 for task 4: output via sigmoid function for binary classification
    """
    def __init__(self):
        super(BinaryNeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=8)
        self.fc5 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        sig = nn.Sigmoid()
        return sig(x)
    
class BinaryNeuralNetSoftmax(nn.Module):
    """
    Variant 2 of model for task 4: output via log softmax for multiclass classification
    """
    def __init__(self):
        super(BinaryNeuralNetSoftmax, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=8)
        self.fc5 = nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        soft = F.log_softmax(x, dim=1)
        return soft

def plot_dataset_distribution(data: np.ndarray,
                          feature_names: list[str],
                          title: str = 'Graph',
                          export: bool = False,
                          export_title: str = 'figexport',
                          show_plot: bool = True):
    """
    Plotting histograms of feature pairs for task 1
    """
    num_features = data.shape[1]
    fig, axes = plt.subplots(int(num_features/2), 2, figsize=(16, 16))
    for i in range(num_features):
        row = int(i % (num_features / 2))
        col = int(i // (num_features / 2))
        axes[row, col].hist(data[:, i], bins=50, color='gray', alpha=0.7)
        axes[row, col].set_xlabel(feature_names[i], size=14)
        axes[row, col].set_ylabel('Abundancy', size=14)
    plt.tight_layout()
    plt.title(title)
    if show_plot:
        plt.show()
    if export:
        os.makedirs('./exports/hist/', exist_ok=True)
        fig.savefig(f"./exports/hist/{export_title}.png",
                    dpi=300)
    plt.close()
    
def plot_dataset_pairs(data_x: np.ndarray,
                             feature_names: list[str],
                             export: bool = False,
                             export_title: str = 'figexport',
                             show_plot: bool = True):
    """
    Scatter plot, iterating through all data pairs
    """
    num_features = data_x.shape[1]
    fig, axes = plt.subplots(num_features, num_features, figsize=(16, 16))
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                axes[i, j].hist(data_x[:, i], bins=50, color='gray', alpha=0.7)
                axes[i, j].set_title(f"{feature_names[i]} Distribution")
            elif i > j:
                axes[i, j].scatter(data_x[:, j], data_x[:, i], alpha=0.5, s=3)
            else:
                axes[i, j].set_visible(False)
            if i == num_features - 1:
                axes[i, j].set_xlabel(feature_names[j])
            if j == 0:
                axes[i, j].set_ylabel(feature_names[i])
    plt.tight_layout()
    plt.title("Histogram: Feature pairs")
    if show_plot:
        plt.show()
    if export:
        os.makedirs('./exports/dataset_pairs', exist_ok=True)
        fig.savefig(f"./exports/dataset_pairs/{export_title}.png",
                    dpi=300)
    plt.close()

def plot_loss_curve(train_loss,
                    val_loss,
                    export_folder="exports",
                    val_loss_label="Validation Loss",
                    export_name="losscurve",
                    show_plot:bool=False):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss)), train_loss, alpha=0.5, color='red', label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss, alpha=0.7, color='blue', label=val_loss_label)
    plt.title('Loss curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if show_plot:
        plt.show()
    os.makedirs(export_folder, exist_ok=True)
    plt.savefig(f"{export_folder}/{export_name}.png", dpi=100)
    plt.close()

def plot_loss_accuracy_curve(train_loss,
                             val_loss,
                             accuracies,
                             export_folder="exports",
                             val_loss_label="Validation Loss",
                             export_name="losscurve",
                             show_plot:bool=False):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss)), train_loss, alpha=0.5, color='red', label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss, alpha=0.7, color='blue', label=val_loss_label)
    plt.plot(range(len(accuracies)), accuracies, alpha=0.7, color='green', label='Accuracy')
    plt.title('Loss & Accuracy curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    if show_plot:
        plt.show()
    os.makedirs(export_folder, exist_ok=True)
    plt.savefig(f"{export_folder}/{export_name}.png", dpi=300)
    plt.close()

def scatter_plot(y_true,
                 y_pred,
                 export_folder="exports/task3_predict/",
                 filename="scatterplot",
                 show_plot:bool=False):
    """
    Scatter plot for plotting predicted vs true y values
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(x=y_pred, y=y_true)
    plt.title("Predicted vs. True House Prices")
    plt.xlabel("Predicted Price [in 100,000 USD]")
    plt.ylabel("True Price [in 100,000 USD]")
    if show_plot:
        plt.show()
    os.makedirs(export_folder, exist_ok=True)
    plt.savefig(f"{export_folder}/{filename}.png", dpi=300)
    plt.close()

def plot_confusion_matrix(y_true,
                          y_pred,
                          export_folder="exports",
                          filename="confusion_matrix",
                          show_plot:bool=False):
    """
    Confusion matrix plot for binary classification task 4
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['< $200k', '>= $200k'],
                yticklabels=['< $200k', '>= $200k'])
    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    if show_plot:
        plt.show()
    plt.tight_layout()
    os.makedirs(export_folder, exist_ok=True)
    plt.savefig(f"{export_folder}/{filename}.png", dpi=300)
    plt.close()


def gather_split_data():
    """
    Downloading/fetching data and splitting it into train, validation and test set
    Returning for fitting, transformation and loading
    """
    X, y = fetch_california_housing(return_X_y=True)
    print(f"Original dataset size: {X.shape[0]}")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=302
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=302
    )
    print(f"Train set size: {X_train.shape[0]}\n")
    print(f"Validation set size: {X_val.shape[0]}\n")
    print(f"Test set size: {X_test.shape[0]}\n")
    return X_train, y_train, X_test, y_test, X_val, y_val


def load_prepare_data(transformed_X_train_split,
                      y_train,
                      transformed_X_val_split,
                      y_val,
                      transformed_X_test_split,
                      y_test):
    
    """
    Converting np arr into tensors and loading into DataLoader for training
    """

    X_train_tensor, y_train_tensor = torch.from_numpy(transformed_X_train_split).float(), torch.from_numpy(y_train).float().reshape(-1, 1)
    # reshape is necessary so that y_train has 2D structure with one column and x rows like x_train has 8 columns and x rows
    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor, y_train_tensor
    )

    X_val_tensor, y_val_tensor = torch.from_numpy(transformed_X_val_split).float(), torch.from_numpy(y_val).float().reshape(-1, 1)
    val_dataset = torch.utils.data.TensorDataset(
        X_val_tensor, y_val_tensor
    )

    X_test_tensor, y_test_tensor = torch.from_numpy(transformed_X_test_split).float(), torch.from_numpy(y_test).float().reshape(-1, 1)
    test_dataset = torch.utils.data.TensorDataset(
        X_test_tensor, y_test_tensor
    )

    training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, generator=g)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, generator=g)

    training_and_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    training_and_val_dataloader = torch.utils.data.DataLoader(
        training_and_val_dataset, batch_size=64, shuffle=True, generator=g
    )
    data = {
        'training_dataloader': training_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'training_and_val_dataloader': training_and_val_dataloader,
        'X_train_tensor': X_train_tensor,
        'X_val_tensor': X_val_tensor,
        'X_test_tensor': X_test_tensor,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    return data

def prepare_data_task4():
    """
    Data preparation specifically for task 4
    Setting y values to class 0 or 1
    """
    X_train, y_train, X_test, y_test, X_val, y_val = gather_split_data()
    scaler = StandardScaler()
    transformed_X_train_split = scaler.fit_transform(X=X_train)
    transformed_X_val_split = scaler.transform(X=X_val)
    transformed_X_test_split = scaler.transform(X=X_test)

    y_train[y_train < 2], y_val[y_val < 2], y_test[y_test < 2] = 0, 0, 0
    y_train[y_train >= 2], y_val[y_val >= 2], y_test[y_test >= 2] = 1, 1, 1

    data = load_prepare_data(transformed_X_train_split=transformed_X_train_split,
                             y_train=y_train,
                             transformed_X_val_split=transformed_X_val_split,
                             y_val=y_val,
                             transformed_X_test_split=transformed_X_test_split,
                             y_test=y_test)
    return data


def task1():
    """
    ### TASK 1
    ### Preparing and loading data, plotting histogram, returning loaded data for task 2, 3 and 4
    """
    print("Executing first task: Data preparation & histogram plotting.")
    X_train, y_train, X_test, y_test, X_val, y_val = gather_split_data()
    scaler = StandardScaler()
    transformed_X_train_split = scaler.fit_transform(X=X_train)
    transformed_X_val_split = scaler.transform(X=X_val)
    transformed_X_test_split = scaler.transform(X=X_test)
        
    plot_dataset_distribution(
        data=X_train,
        feature_names=feature_names,
        title='Training set data distribution',
        export=True,
        export_title='task1_training_set_distribution',
        show_plot=False,
    )

    plot_dataset_distribution(
        data=transformed_X_train_split,
        feature_names=feature_names,
        title='Fit transformed training data',
        export=True,
        export_title='task1_training_set_distrib_fit_transformed',
        show_plot=False
    )

    plot_dataset_distribution(
        data=transformed_X_val_split,
        feature_names=feature_names,
        title='Fit transformed validation data',
        export=True,
        export_title='task1_val_set_distrib_fit_transformed',
        show_plot=False
    )

    plot_dataset_pairs(
        data_x = X_train,
        feature_names=feature_names,
        export=True,
        show_plot=False
    )
    #training and val dataloader is used for task 3
    data = load_prepare_data(transformed_X_train_split=transformed_X_train_split,
                             y_train=y_train,
                             transformed_X_val_split=transformed_X_val_split,
                             y_val=y_val,
                             transformed_X_test_split=transformed_X_test_split,
                             y_test=y_test)
                                                                                                       
    return data


def task2(training_dataloader, val_dataloader):
    """
    Training different model architectures w/ different hyperparameters
    Export of important output values to xlsx file for comparison of model architectures and hyperp.
    """
    print("Executing task 2: Training different model arch and hyperparams. Export of output to xlsx file.")
    df_data = []
    for i, layersize in enumerate(TASK2_MODEL_ARCHITECTURES):
        for j, learning_rate in enumerate(TASK2_HYPERPARAMETERS['learning_rate']):
            for k, epoch_num in enumerate(TASK2_HYPERPARAMETERS['epochs']):
                print(f"Arch {i}, Learning Rate {j}, Epoch Var {k}")
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)
                model_arch = NeuralNet(layer_sizes=layersize)
                train_loss, val_loss = train_eval_model(
                train_loader=training_dataloader,
                val_loader=val_dataloader,
                model=model_arch,
                epochs=epoch_num,
                learning_rate=learning_rate)
                losses = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epochs': [i for i in range (1, epoch_num+1)]
                }
                new_df = pd.DataFrame(losses)
                folder = f"./exports/losses/{str(layersize[1:-1])}_{str(learning_rate)}_{str(epoch_num)}/"
                os.makedirs(folder, exist_ok=True)
                new_df.to_excel(f"{folder}/loss_data.xlsx")
                best_train_row = new_df.nsmallest(1, 'train_loss').iloc[0]
                best_val_row   = new_df.nsmallest(1, 'val_loss').iloc[0]
                data = {
                    'HL': len(model_arch.layers) - 1,
                    'HU': model_arch.layer_sizes[1:-1],
                    'Final train loss': f"{train_loss[-1]:.3f}",
                    'Best train loss': f"{best_train_row['train_loss']:.3f}",
                    'Best train loss epoch': int(best_train_row['epochs']),
                    'Final val loss': f"{val_loss[-1]:.3f}",
                    'Best val loss': f"{best_val_row['val_loss']:.3f}",
                    'Best val loss epoch': int(best_val_row['epochs']),
                    'Learning Rate': learning_rate,
                    'Total epochs': epoch_num
                }
                df_data.append(data)
                print(f"Final Training Loss: {train_loss[-1]:.3f}\n")
                print(f"Final Val Loss: {val_loss[-1]:.3f}\n")
                print(f"Best Training Loss: {best_train_row['train_loss']:.3f} | Epoch: {best_train_row['epochs']:.3f}\n")
                print(f"Best Val Loss: {best_val_row['val_loss']:.3f} | Epoch: {best_val_row['epochs']:.3f}\n")
                plot_loss_curve(train_loss=train_loss, val_loss=val_loss, export_folder=folder)
    df = pd.DataFrame(df_data)
    df.to_excel("exports/losses/training_model_variation_test.xlsx")


def task3(training_and_val_dataloader, test_dataloader):
    """
    Final training of model w/ selected arch and hyperp.
    Returning model for prediction on test dataset
    """
    print("Executing task 3: Final training with selected model architecture.")
    selected_architecture = TASK3_FINAL_MODEL_ARCHITECTURE
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = NeuralNet(selected_architecture)
    lr = TASK3_LR
    epoch_num = TASK3_EPOCHS

    train_loss, test_loss = train_eval_model(
                train_loader=training_and_val_dataloader,
                val_loader=test_dataloader,
                model=model,
                epochs=epoch_num,
                learning_rate=lr)

    losses = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epochs': [i for i in range (1, epoch_num+1)]
        }
    df = pd.DataFrame(losses)
    
    new_df = pd.DataFrame(losses)
    best_train_row = new_df.nsmallest(1, 'train_loss').iloc[0]
    best_test_row   = new_df.nsmallest(1, 'test_loss').iloc[0]
    print("-"*30)
    print(f"Final Training Loss: {train_loss[-1]:.3f}\n")
    print(f"Final Test Loss: {test_loss[-1]:.3f}\n")
    print(f"Best Training Loss: {best_train_row['train_loss']:.3f} | Epoch: {best_train_row['epochs']:.3f}\n")
    print(f"Best Test Loss: {best_test_row['test_loss']:.3f} | Epoch: {best_test_row['epochs']:.3f}\n")
    print("-"*30)
    plot_loss_curve(
        train_loss=train_loss,
        val_loss=test_loss,
        export_folder="exports/task3_final_training/",
        val_loss_label="Test loss",
        export_name="task3_testset_losscurve"
    )

    df.to_excel("exports/task3_final_training/losses.xlsx")
    return model

def task3_predict(model, X_test_tensor, y_true):
    print("Executing task 3 prediction: Prediction on test data.")
    model.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        prediction_array = model(X_test_tensor)
        scatter_plot(y_true=y_true,
                     y_pred=prediction_array,
                     export_folder="exports/task3_predict")

def task4(training_val_dataloader, test_dataloader, X_test_tensor, y_test, lossfunc='bceloss'):
    """
    Training binary / multiclass classification model
    Plotting loss curve + confusion matrix
    """
    print("Executing task 4: Classification.")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if lossfunc == 'bceloss':
        model = BinaryNeuralNet()
    elif lossfunc == 'nllloss':
        model = BinaryNeuralNetSoftmax()
    lr = TASK4_LEARNING_RATE
    epochs = TASK4_EPOCHS

    train_loss, val_loss, acc = train_eval_model_binary(
        train_loader=training_val_dataloader,
        test_loader=test_dataloader,
        model=model,
        epochs=epochs,
        learning_rate=lr,
        lossfunc=lossfunc
    )
    losses = {
            'train_loss': train_loss,
            'test_loss': val_loss,
            'epochs': [i for i in range (1, epochs+1)]
        }
    
    new_df = pd.DataFrame(losses)
    best_train_row = new_df.nsmallest(1, 'train_loss').iloc[0]
    best_test_row   = new_df.nsmallest(1, 'test_loss').iloc[0]
    print("-"*30)
    print(f"Final Training Loss: {train_loss[-1]:.3f}\n")
    print(f"Final Test Loss: {val_loss[-1]:.3f}\n")
    print(f"Best Training Loss: {best_train_row['train_loss']:.3f} | Epoch: {best_train_row['epochs']:.3f}\n")
    print(f"Best Test Loss: {best_test_row['test_loss']:.3f} | Epoch: {best_test_row['epochs']:.3f}\n")
    print("-"*30)

    plot_loss_accuracy_curve(train_loss=train_loss,
                             val_loss=val_loss,
                             accuracies=acc,
                             export_folder=f"exports/task4_{lossfunc}",
                             val_loss_label="Test Loss",
                             export_name=f"loss_accuracy_{lossfunc}",
                             show_plot=False)
    model.eval()
    with torch.no_grad():
        y_hat_prob = model(X_test_tensor).numpy()
        if lossfunc == 'nllloss':
            y_pred = y_hat_prob.argmax(axis=1)
        elif lossfunc == 'bceloss':
            y_pred = (y_hat_prob >= 0.5).astype(float)
    plot_confusion_matrix(y_true=y_test.reshape(-1, 1),
                          y_pred=y_pred,
                          export_folder=f"exports/task4_{lossfunc}",
                          filename="confusion_matrix")
    

def main():
    data = task1()
    task2(training_dataloader=data['training_dataloader'], val_dataloader=data['val_dataloader'])
    final_trained_model = task3(training_and_val_dataloader=data['training_and_val_dataloader'],
                                            test_dataloader=data['test_dataloader'])
    task3_predict(model=final_trained_model,
                X_test_tensor=data['X_test_tensor'],
                y_true=data['y_test'])

    data_task4 = prepare_data_task4()

    print(f"Training dataset size: {len(data_task4['training_and_val_dataloader'].dataset)}")
    print(f"Test dataset size: {len(data_task4['test_dataloader'].dataset)}")

    #training with NLLLoss func.
    task4(training_val_dataloader=data_task4['training_and_val_dataloader'],
        test_dataloader=data_task4['test_dataloader'],
        X_test_tensor=data_task4['X_test_tensor'],
        y_test=data_task4['y_test'],
        lossfunc='nllloss')

    #training with BCELoss func.
    task4(training_val_dataloader=data_task4['training_and_val_dataloader'],
        test_dataloader=data_task4['test_dataloader'],
        X_test_tensor=data_task4['X_test_tensor'],
        y_test=data_task4['y_test'],
        lossfunc='bceloss')


if __name__ == '__main__':
    main()