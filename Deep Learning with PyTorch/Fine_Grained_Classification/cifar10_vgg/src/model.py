import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import pytorch_lightning as pl
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, parameters):
        super(CIFAR10Classifier, self).__init__()
        self.pretrained_model_name = parameters['pretrained_model_name']
        self.pretrained_model_path = parameters['pretrained_model_path']
        self.num_classes = parameters['num_classes']
        self.batch_size = parameters['batch_size']
        self.resizing_factor = parameters['resizing_factor']
        self.loss_fn = nn.NLLLoss()
        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': []}
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
            self.model.load_state_dict(torch.load(
                self.pretrained_model_path, map_location=torch.device('cpu')))

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

        n = n*2  # since weights and bias are included as separate
        total_layers = len(list(self.model.classifier.parameters()))

        # invalid n
        if n > total_layers:
            print(
                f"Warning: There are only {total_layers} layers in the model. Cannot unfreeze {n} layers.")

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

        n = n*2  # since weights and bias are included as separate
        total_layers = len(list(self.model.features.parameters()))

        # invalid n
        if n > total_layers:
            print(
                f"Warning: There are only {total_layers} layers in the model. Cannot unfreeze {n} layers.")
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
        self.logger.experiment.add_scalar(
            'train_loss', avg_epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            'train_acc', avg_epoch_loss, self.current_epoch)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item()/self.batch_size
        self.validation_step_outputs.append((loss.item(), acc))
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
        self.logger.experiment.add_scalar(
            'val_loss', avg_epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            'val_acc', avg_epoch_loss, self.current_epoch)
        self.validation_step_outputs.clear()

    def test_step(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad,
                         self.model.parameters()), lr=0.001)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.resizing_factor),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar10_train = datasets.CIFAR10(
            root='./data', train=True, transform=transform, download=True)
        return DataLoader(cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.resizing_factor),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar10_val = datasets.CIFAR10(
            root='./data', train=False, transform=transform, download=True)
        return DataLoader(cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.resizing_factor),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        cifar10_test = datasets.CIFAR10(
            root='./data', train=False, transform=transform, download=True)
        return DataLoader(cifar10_test, batch_size=self.batch_size)

    def get_history(self):
        return self.history

    @logger.setter
    def logger(self, value):
        self._logger = value
