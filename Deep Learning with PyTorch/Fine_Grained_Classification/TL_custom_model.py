import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import datasets, transforms, models
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split
from IPython.core.display import set_matplotlib_formats


def save_plots(history, save_dir, model_name, batch_size, training_type):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    # create train_loss vs. val_loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.title('Training Vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    name = str(save_dir)+'/'+'loss_'+str(model_name)+'_'+str(batch_size)+'_'+str(training_type)
    plt.savefig(name)

    # create train_acc vs. val_acc
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.title('Training Vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    name = str(save_dir)+'/'+'loss_'+str(model_name)+'_'+str(batch_size)+'_'+str(training_type)
    plt.savefig(name)
    
    
class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, pretrained_model_name, pretrained_model_path, num_classes, batch_size, resizing_factor):
        super(CIFAR10Classifier, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_path = pretrained_model_path
        self.num_classes = num_classes
        self.loss_fn = nn.NLLLoss()
        self.batch_size = batch_size
        self.resizing_factor = resizing_factor
        self.history = {'train_loss': [], 'train_acc':[], 'val_loss': [], 'val_acc':[]}
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.logger = None

        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # transfer learning parameters
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.classifiers_n = -1
        self.features_n = -1

        # check for GPU availability
        use_gpu = torch.cuda.is_available()

        # load model architectures without weight
        if use_gpu:
            self.model = getattr(models, self.pretrained_model_name)().cuda()
        else:
            self.model = getattr(models, self.pretrained_model_name)()

        # load pre-trained weights
        if use_gpu:
            self.model.load_state_dict(torch.load(self.pretrained_model_path))
        else:
            self.model.load_state_dict(torch.load(self.pretrained_model_path, map_location=torch.device('cpu')))

        # get input dimension of the fc layer to be replaced and index of the last fc layer
        self.in_feat = self.model.classifier[-1].in_features
        fc_idx = len(self.model.classifier) - 1

        custom_fc = nn.Sequential(nn.Linear(self.in_feat, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, self.num_classes),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.LogSoftmax(dim=1))

        # add custom fc layers to model
        self.model.classifier[fc_idx] = custom_fc

    def on_fit_start(self):
        self.logger = self.trainer.logger

    def forward(self, x):
        x = self.model(x)
        return x

    # freezes all layers in the model
    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # unfreeze last 'n' fully connected layers
    def unfreeze_last_n_fc_layers(self, n):

        # if n == -1 don't unfreeze any layers
        if n == -1:
            return 0

        n = n*2 # since weights and bias are included as separate
        total_layers = len(list(self.model.classifier.parameters()))

        # invalid n
        if n > total_layers:
            print(f"Warning: There are only {total_layers} layers in the model. Cannot unfreeze {n} layers.")

        # if n == 0 unfreeze all layers
        elif n == 0:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            for i, param in enumerate(self.model.classifier.parameters()):
                if i >= (total_layers - n):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    # unfreeze last 'n' fully connected layers
    def unfreeze_last_n_conv_layers(self, n):

        # if n == -1 don't unfreeze any layers
        if n == -1:
            return 0

        n = n*2 # since weights and bias are included as separate
        total_layers = len(list(self.model.features.parameters()))

        # invalid n
        if n > total_layers:
            print(f"Warning: There are only {total_layers} layers in the model. Cannot unfreeze {n} layers.")
        # if n == 0 unfreeze all layers
        elif n == 0:
            for param in self.model.features.parameters():
                param.requires_grad = True
        else:
            for i, param in enumerate(self.model.features.parameters()):
                if i >= total_layers - n:
                    param.requires_grad = True
                else:
                    pass

    # set parameters for transfer learning
    def set_transfer_learning_params(self, unfreeze_n_fc, unfreeze_n_conv):
        self.classifier_n = unfreeze_n_fc
        self.features_n = unfreeze_n_conv
        self.freeze_all_layers()
        self.unfreeze_last_n_fc_layers(unfreeze_n_fc)
        self.unfreeze_last_n_conv_layers(unfreeze_n_conv)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item()/self.batch_size
        self.training_step_outputs.append((loss.item(), acc))
        #self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        num_items = len(self.training_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.training_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss/num_items
        avg_epoch_acc = cum_acc/num_items
        self.history['train_loss'].append(avg_epoch_loss)
        self.history['train_acc'].append(avg_epoch_acc)
        self.logger.experiment.add_scalar('train_loss', avg_epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar('train_acc', avg_epoch_loss, self.current_epoch)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item()/self.batch_size
        self.validation_step_outputs.append((loss.item(), acc))
        #self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        num_items = len(self.validation_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.validation_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss/num_items
        avg_epoch_acc = cum_acc/num_items
        self.history['val_loss'].append(avg_epoch_loss)
        self.history['val_acc'].append(avg_epoch_acc)
        self.logger.experiment.add_scalar('val_loss', avg_epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar('val_acc', avg_epoch_loss, self.current_epoch)
        self.validation_step_outputs.clear()

    def test_step(self):
        pass

    def on_test_epoch_end(self):
        pass
        
    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p:p.requires_grad, self.model.parameters()), lr=0.001)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.resizing_factor),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar10_train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        return DataLoader(cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.resizing_factor),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar10_val = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        return DataLoader(cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.resizing_factor),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar10_test = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        return DataLoader(cifar10_test, batch_size=self.batch_size)

    def get_history(self):
        return self.history
    
    
def train_custom_fc_layers(model, epochs, unfreeze_n_fc, log_dir, plots_dir):
    # freeze all layers except the last two fc layers
    unfreeze_n_conv = -1
    model.set_transfer_learning_params(unfreeze_n_fc, unfreeze_n_conv)

    # initialize logger
    model_name = str(model.pretrained_model_name)
    training_type = 'fc'
    batch_size = str(model.batch_size)
    log_dir = log_dir + model_name + '/' + training_type + '/batchsz' + batch_size

    # create directories if non-existent
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {log_dir}: {e}")

    # initalize logger
    logger = TensorBoardLogger(log_dir)
    logger.log_hyperparams({'epochs': epochs,
                                'batch_size': model.batch_size,
                                'name': model_name})

    # train model
    trainer = pl.Trainer(max_epochs = epochs, logger = logger)
    trainer.fit(model)
    print(f'Training Complete. Results logges at {log_dir}')

    # get training history
    history = model.get_history()

    # plot history
    save_plots(history, plots_dir, model_name, model.batch_size, 'fc')

    return model

def train_entire_fc_block(model, epochs, unfreeze_n_fc = 0, log_dir=None, plots_dir=None):
    # freeze all layers except the fc block
    unfreeze_n_conv = -1
    model.set_transfer_learning_params(unfreeze_n_fc, unfreeze_n_conv)

    # initialize logger
    model_name = str(model.pretrained_model_name)
    training_type = 'compfc'
    batch_size = str(model.batch_size)
    log_dir = log_dir + model_name + '/' + training_type + '/batchsz' + batch_size

    # create directories if non-existent
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {log_dir}: {e}")

    # initalize logger
    logger = TensorBoardLogger(log_dir)
    logger.log_hyperparams({'epochs': epochs,
                                'batch_size': model.batch_size,
                                'name': model_name})

    # train model
    trainer = pl.Trainer(max_epochs = epochs, logger = logger)
    trainer.fit(model)
    print(f'Training Complete. Results logges at {log_dir}')

    # get training history
    history = model.get_history()

    # save history
    save_plots(history, plots_dir, model_name, model.batch_size, 'compfc')

    return model

def train_conv_layers(model, epochs, unfreeze_n_conv, log_dir, plots_dir):
    # freeze all layers except the last two conv layers
    unfreeze_n_fc = -1
    model.set_transfer_learning_params(unfreeze_n_fc, unfreeze_n_conv)

    # initialize logger
    model_name = str(model.pretrained_model_name)
    training_type = 'conv'
    batch_size = str(model.batch_size)
    log_dir = log_dir + model_name + '/' + training_type + '/batchsz' + batch_size

    # create directories if non-existent
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {log_dir}: {e}")

    # initalize logger
    logger = TensorBoardLogger(log_dir)
    logger.log_hyperparams({'epochs': epochs,
                                'batch_size': model.batch_size,
                                'name': model_name})

    # train model
    trainer = pl.Trainer(max_epochs = epochs, logger = logger)
    trainer.fit(model)
    print(f'Training Complete. Results logges at {log_dir}')

    # get training history
    history = model.get_history()

    # save history
    save_plots(history, plots_dir, model_name, model.batch_size, 'conv')

    return model

def fine_tune_model(model, epochs, unfreeze_n_fc = 0, unfreeze_n_conv = 2, log_dir=None, plots_dir=None):
    # freeze all layers except the last two conv layers and the fc block
    model.set_transfer_learning_params(unfreeze_n_fc, unfreeze_n_conv)

    # initialize logger
    model_name = str(model.pretrained_model_name)
    training_type = 'finetuning'
    batch_size = str(model.batch_size)
    log_dir = log_dir + model_name + '/' + training_type + '/batchsz' + batch_size

    # create directories if non-existent
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {log_dir}: {e}")

    # initalize logger
    logger = TensorBoardLogger(log_dir)
    logger.log_hyperparams({'epochs': epochs,
                                'batch_size': model.batch_size,
                                'name': model_name})

    # train model
    trainer = pl.Trainer(max_epochs = epochs, logger = logger)
    trainer.fit(model)
    print(f'Training Complete. Results logges at {log_dir}')

    # get training history
    history = model.get_history()

    # save history
    save_plots(history, plots_dir, model_name, model.batch_size, 'finetune')

    return model


def TransferLearningVGG(n_fc, n_compfc, n_conv, n_ft_fc, n_ft_conv, epochs_fc, epochs_compfc,
                                  epochs_conv, epochs_finetune, batch_size):
    # define variables
    pretrained_model_name = 'vgg16'
    pretrained_model_path = 'pretrained_models/vgg16.pth'
    log_dir = 'logs/'
    plots_dir = 'plots/'
    num_classes_CIFAR10 = 10
    resizing_factor_VGG = (224, 224)

    # initialize model
    custom_model = CIFAR10Classifier(pretrained_model_name, pretrained_model_path, num_classes_CIFAR10,
                                     batch_size, resizing_factor_VGG)

    # print model architecture
    print(custom_model.model)

    # transfer learning steps
    # 1. Train added fc layers
    n_fc = 2 # fc layers to unfreeze from last
    custom_model = train_custom_fc_layers(custom_model, epochs_fc, n_fc, log_dir, plots_dir)
    

    '''
    # 2. Train all fc layers
    n_compfc = 0 # fc layers to unfreeze from last; 0 coresponds to all
    compfc_history = train_entire_fc_block(custom_model, epochs_compfc, n_compfc)
    plot_history(compfc_history)

    # 3. Train convolutional layers
    n_conv = 2
    conv_history = train_conv_layers(custom_model, epochs_conv, n_conv)
    plot_history(conv_history)

    # 4. Fine tune model
    n_ft_fc = 0    # no. of fc layers to unfreeze
    n_ft_conv = 2  # no. of conv layers to unfreeze
    finetune_history = fine_tune_model(custom_model, epochs_finetune, n_ft_fc, n_ft_conv)
    plot_history(finetune_history)
    '''

    return custom_model

# 1. added fully-connected layers training
n_fc = 2 # fc layers to unfreeze from last
epochs_fc = 8

# 2. entire fully-connected block training
n_compfc = 0 # fc layers to unfreeze from last; 0 coresponds to all
epochs_compfc = 8

# 3. convolutional layers training
n_conv = 2 # no. of convolutional layers to unfreeze from last
epochs_conv = 10

# 4. fine-tuning
n_ft_fc = 0    # no. of fully connected layers to unfreeze
n_ft_conv = 2  # no. of convolutional layers to unfreeze
epochs_finetune = 10

batch_size = 64

# start training pipeling
final_model = TransferLearningVGG(n_fc, n_compfc, n_conv, n_ft_fc, n_ft_conv, epochs_fc, epochs_compfc,
                                  epochs_conv, epochs_finetune, batch_size)
