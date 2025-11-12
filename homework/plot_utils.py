import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_dataset_distribution(data: np.ndarray,
                          feature_names: list[str],
                          title: str = 'Graph',
                          export: bool = False,
                          export_title: str = 'figexport',
                          show_plot: bool = True):
    num_features = data.shape[1]
    fig, axes = plt.subplots(int(num_features/2), 2, figsize=(16, 16))
    for i in range(num_features):
        row = int(i % (num_features / 2))
        col = int(i // (num_features / 2))
        axes[row, col].hist(data[:, i], bins=50, color='gray', alpha=0.7)
        axes[row, col].set_xlabel(feature_names[i], size=14)
        axes[row, col].set_ylabel('Abundancy', size=14)
    plt.tight_layout()
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
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred, y_true)
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
