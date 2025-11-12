from plot_utils import plot_loss_curve, plot_dataset_pairs, plot_dataset_distribution, scatter_plot
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import NeuralNet, BinaryNeuralNet, BinaryNeuralNetSoftmax
from train import train_eval_model, train_eval_model_binary
from plot_utils import plot_loss_accuracy_curve, plot_confusion_matrix
import pandas as pd
import numpy as np
import os
torch.manual_seed(302)
np.random.seed(302)

g = torch.Generator()
g.manual_seed(302)

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

def gather_split_data():
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


### TASK 1
### Preparing and loading data, plotting histogram, returning loaded data
def task1():
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
        show_plot=False
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
        title='Fit transformed training data',
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

### TASK 2
### Iterating through different model architecture designs and hyperparameters
def task2(training_dataloader, val_dataloader):


    df_data = []
    for i, layersize in enumerate(TASK2_MODEL_ARCHITECTURES):
        for j, learning_rate in enumerate(TASK2_HYPERPARAMETERS['learning_rate']):
            for k, epoch_num in enumerate(TASK2_HYPERPARAMETERS['epochs']):
                print(f"Arch {i}, Learning Rate {j}, Epoch Var {k}")
                SEED = 302
                torch.manual_seed(SEED)
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
                    'Final train loss': f"{train_loss[-1]:.4f}",
                    'Best train loss': f"{best_train_row['train_loss']:.4f}",
                    'Best train loss epoch': int(best_train_row['epochs']),
                    'Final val loss': f"{val_loss[-1]:.4f}",
                    'Best val loss': f"{best_val_row['val_loss']:.4f}",
                    'Best val loss epoch': int(best_val_row['epochs']),
                    'Learning Rate': learning_rate,
                    'Total epochs': epoch_num
                }
                df_data.append(data)
                print(f"Final Training Loss: {train_loss[-1]}\n")
                print(f"Final Val Loss: {val_loss[-1]}\n")
                print(f"Best Training Loss: {best_train_row['train_loss']} | Epoch: {best_train_row['epochs']}\n")
                print(f"Best Val Loss: {best_val_row['val_loss']} | Epoch: {best_val_row['epochs']}\n")
                plot_loss_curve(train_loss=train_loss, val_loss=val_loss, export_folder=folder)
    df = pd.DataFrame(df_data)
    df.to_excel("exports/losses/training_model_variation_test.xlsx")

def task3(training_dataloader, val_dataloader):
    selected_architecture = TASK3_FINAL_MODEL_ARCHITECTURE
    SEED = 302
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = NeuralNet(selected_architecture)
    lr = TASK3_LR
    epoch_num = TASK3_EPOCHS
    train_loss, val_loss = train_eval_model(
                train_loader=training_dataloader,
                val_loader=val_dataloader,
                model=model,
                epochs=epoch_num,
                learning_rate=lr)
    losses = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epochs': [i for i in range (1, epoch_num+1)]
        }
    new_df = pd.DataFrame(losses)
    best_train_row = new_df.nsmallest(1, 'train_loss').iloc[0]
    best_val_row = new_df.nsmallest(1, 'val_loss').iloc[0]

    plot_loss_curve(
        train_loss=train_loss,
        val_loss=val_loss,
        export_folder=f"exports/task3/",
        export_name="task3_losscurve"
    )
    data = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epochs': [i for i in range(1, epoch_num+1)]
    }
    print("-"*30)
    print(f"Last Train loss: {train_loss[-1]} and last Val Loss: {val_loss[-1]} | Epoch {epoch_num}\n")
    print(f"Best Training Loss: {best_train_row['train_loss']} | Epoch: {best_train_row['epochs']}\n")
    print(f"Best Val Loss: {best_val_row['val_loss']} | Epoch: {best_val_row['epochs']}\n")
    print("-"*30)
    df = pd.DataFrame(data)
    df.to_excel("exports/task3/chosen_model_plot_data.xlsx")
    return model

def task3_final_training(training_and_val_dataloader, test_dataloader):
    selected_architecture = TASK3_FINAL_MODEL_ARCHITECTURE
    SEED = 302
    torch.manual_seed(SEED)
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
    print(f"Final Training Loss: {train_loss[-1]}\n")
    print(f"Final Test Loss: {test_loss[-1]}\n")
    print(f"Best Training Loss: {best_train_row['train_loss']} | Epoch: {best_train_row['epochs']}\n")
    print(f"Best Test Loss: {best_test_row['test_loss']} | Epoch: {best_test_row['epochs']}\n")
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
    model.eval()
    with torch.no_grad():
        prediction_array = model(X_test_tensor)
        scatter_plot(y_true=y_true,
                     y_pred=prediction_array,
                     export_folder="exports/task3_predict")

def task4(training_val_dataloader, test_dataloader, X_test_tensor, y_test, lossfunc='bceloss'):
    SEED = 302
    torch.manual_seed(SEED)
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
    print(f"Final Training Loss: {train_loss[-1]}\n")
    print(f"Final Test Loss: {val_loss[-1]}\n")
    print(f"Best Training Loss: {best_train_row['train_loss']} | Epoch: {best_train_row['epochs']}\n")
    print(f"Best Test Loss: {best_test_row['test_loss']} | Epoch: {best_test_row['epochs']}\n")
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
    




#data = task1()
#task2(training_dataloader=data['training_dataloader'], val_dataloader=data['val_dataloader'])
#task3(training_dataloader=data['training_dataloader'], val_dataloader=data['val_dataloader'])
#final_trained_model = task3_final_training(training_and_val_dataloader=data['training_and_val_dataloader'],
                                           #test_dataloader=data['test_dataloader'])
#task3_predict(model=final_trained_model, X_test_tensor=data['X_test_tensor'], y_true=data['y_test'])

data_task4 = prepare_data_task4()

print(f"Training dataset size: {len(data_task4['training_and_val_dataloader'].dataset)}")
print(f"Test dataset size: {len(data_task4['test_dataloader'].dataset)}")

task4(training_val_dataloader=data_task4['training_and_val_dataloader'],
      test_dataloader=data_task4['test_dataloader'],
      X_test_tensor=data_task4['X_test_tensor'],
      y_test=data_task4['y_test'],
      lossfunc='nllloss')

task4(training_val_dataloader=data_task4['training_and_val_dataloader'],
      test_dataloader=data_task4['test_dataloader'],
      X_test_tensor=data_task4['X_test_tensor'],
      y_test=data_task4['y_test'],
      lossfunc='bceloss')




### TASK 4
