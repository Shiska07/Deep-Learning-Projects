import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def load_parameters(json_file):
    try:
        with open(json_file, 'r') as file:
            parameters = json.load(file)
        return parameters
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON file '{json_file}' is not a valid JSON file.")
        return None

def save_history(history, history_dir, model_name, batch_size, training_type):

    history_file_path =history_dir + \
        str(model_name) + '/batchsz' + str(batch_size) + '/' + str(training_type)

    # create directory if non-existent
    try:
        os.makedirs(history_file_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {history_file_path}: {e}")


    # create and save df
    file_path = os.path.join(history_file_path, f'history.csv')
    if os.path.isfile(file_path):
        os.remove(file_path)
    df = pd.DataFrame(history)
    df.to_csv(file_path, index = False)

    return file_path


def save_hyperparams(history_dir, h_params):

    # save hyperpapramters dictionary as a .json file
    file_path = os.path.join(history_dir, 'hyperparameters.json')
    with open(file_path, 'w') as json_file:
        json.dump(h_params, json_file)

def save_plots(history, plots_dir, model_name, batch_size, training_type):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    plots_file_path = plots_dir + \
        str(model_name) + '/batchsz' + str(batch_size) + '/' + str(training_type)
    # create directory if non-existent
    try:
        os.makedirs(plots_file_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {plots_file_path}: {e}")

    # create train_loss vs. val_loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.title('Training Vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    name = os.path.join(plots_file_path, 'loss.jpeg')
    if os.path.isfile(name):
        os.remove(name)
    plt.savefig(name)

    # create train_acc vs. val_acc
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.title('Training Vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    name = os.path.join(plots_file_path, 'acc.jpeg')
    if os.path.isfile(name):
        os.remove(name)
    plt.savefig(name)

    return plots_file_path
