import os
import json
import pandas as pd
import matplotlib.pyplot as plt

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

def save_history(history, history_dir, model_name, params_fname, batch_size, training_type, h_params):

    history_file_path = os.path.join(history_dir,
        str(model_name), str(params_fname), str(batch_size), str(training_type))

    # create directory if non-existent
    try:
        os.makedirs(history_file_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {history_file_path}: {e}")


    # create and save df
    csv_file_path = os.path.join(history_file_path, f'history{training_type}.csv')
    if os.path.isfile(csv_file_path):
        os.remove(csv_file_path)
    df = pd.DataFrame(history)
    df.to_csv(csv_file_path, index = False)

    hp_file_path = os.path.join(history_file_path, f'hyperparameters{training_type}.json')
    with open(hp_file_path, 'w') as json_file:
        json.dump(h_params, json_file)

    return history_file_path


def save_plots(history, plots_dir, model_name, params_fname, batch_size, training_type):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    plots_file_path = os.path.join(plots_dir,
        str(model_name), str(params_fname), str(batch_size), str(training_type))
    # create directory if non-existent
    try:
        os.makedirs(plots_file_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {plots_file_path}: {e}")

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    # create train_loss vs. val_loss
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.title(f'Training Vs Validation Loss {training_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # create train_acc vs. val_acc
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.title(f'Training Vs Validation Accuracy {training_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    name = os.path.join(plots_file_path, 'acc_and_loss.jpeg')
    if os.path.isfile(name):
        os.remove(name)
    plt.savefig(name)

    return plots_file_path
