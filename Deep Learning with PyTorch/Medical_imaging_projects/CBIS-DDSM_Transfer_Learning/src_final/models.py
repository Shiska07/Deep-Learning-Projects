import os
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import pytorch_lightning as pl
from custom_layers import get_layers
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


class CBISDDSMPatchClassifierVGG(pl.LightningModule):
    def __init__(self, parameters):
        super(CBISDDSMPatchClassifierVGG, self).__init__()
        self.pretrained_model_name = 'vgg19'
        self.pretrained_model_path = parameters['pretrained_model_path']
        self.params_filename = parameters['filename']
        self.batch_size = parameters['batch_size']
        self.modification_type = parameters['modification_type']
        self.model_dest_folder = parameters['model_dst']
        self.num_classes = parameters['num_classes']
        self.loss_fn = nn.NLLLoss()

        # transfer learning parameters
        self.classifiers_n = -1
        self.features_n = -1

        # activation maps parameters
        self.conv_layer_numbers = [34, 32, 30, 28, 25, 23, 21, 19, 16]

        # store learning rates for different layers
        self.lr = dict()
        self.lr['init'] = parameters['lr_init']
        self.lr['fc'] = parameters['lr_fc']
        self.lr['conv'] = parameters['lr_conv']

        # lists to store outputs from each train/val step
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': []}
        self.test_history = {'test_acc': [], 'test_loss': []}

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

        custom_layers = get_layers(self.pretrained_model_name, self.num_classes, self.modification_type)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = custom_layers['final_classifier']

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self, mode=None):
        optim_params = []

        # new arc modification
        params = self.model.classifier.parameters()
        optim_params.append({'params': params, 'lr': self.lr['fc']})

        for i, layer_number in enumerate(self.conv_layer_numbers):
            params = self.model.features[layer_number].parameters()
            optim_params.append({'params': params, 'lr': self.lr['fc'] / (2 * (i + 1))})

        optimizer = Adam(optim_params)
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Group {i + 1} - Learning Rate: {param_group['lr']}")

        return optimizer

    # freezes all layers in the model
    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # unfreeze last 'n' fully connected layers
    def unfreeze_last_n_fc_layers(self, n):

        # if n == -1 don't unfreeze any layers
        if n == -1:
            return 0

        n = n * 2  # since weights and bias are included as separate
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

        n = n * 2  # since weights and bias are included as separate
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
        self.classifiers_n = unfreeze_n_fc
        self.features_n = unfreeze_n_conv
        self.freeze_all_layers()
        self.unfreeze_last_n_fc_layers(unfreeze_n_fc)
        self.unfreeze_last_n_conv_layers(unfreeze_n_conv)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.training_step_outputs.append((loss.item(), acc))
        return loss

    def on_train_epoch_end(self):
        num_items = len(self.training_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.training_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss / num_items
        avg_epoch_acc = cum_acc / num_items
        self.history['train_loss'].append(avg_epoch_loss)
        self.history['train_acc'].append(avg_epoch_acc)
        print(f'\nTraining Epoch({self.current_epoch}): loss: {avg_epoch_loss}, acc:{avg_epoch_acc}')
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.validation_step_outputs.append((loss.item(), acc))
        return loss

    def on_validation_epoch_end(self):
        num_items = len(self.validation_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.validation_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss / num_items
        avg_epoch_acc = cum_acc / num_items
        self.history['val_loss'].append(avg_epoch_loss)
        self.history['val_acc'].append(avg_epoch_acc)
        print(f'\nValidation Epoch({self.current_epoch}): loss: {avg_epoch_loss}, acc:{avg_epoch_acc}')
        self.validation_step_outputs.clear()
        self.log("val_loss", avg_epoch_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.test_step_outputs.append((loss.item(), acc))
        return loss

    def on_test_epoch_end(self):
        num_items = len(self.test_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.test_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss / num_items
        avg_epoch_acc = cum_acc / num_items
        print(f'Test Epoch loss: {avg_epoch_loss} Test epoch Acc: {avg_epoch_acc}')
        self.test_history['test_loss'].append(avg_epoch_loss)
        self.test_history['test_acc'].append(avg_epoch_acc)
        self.test_step_outputs.clear()

    def get_history(self):
        # remove the first validation epoch data
        self.history['val_loss'].pop(0)
        self.history['val_acc'].pop(0)
        return self.history

    def get_test_history(self):
        return self.test_history

    def clear_history(self):
        for key in self.history:
            self.history[key] = []

    def save_model(self, ckpt_path):
        # save the entire model
        model_architecture_path = os.path.join(ckpt_path, 'arc.pth')
        model_weights_path = os.path.join(ckpt_path, 'weights.pth')
        torch.save(self.model, model_architecture_path)
        torch.save(self.model.state_dict(), model_weights_path)
        print(f'Model saved at {ckpt_path}')


class CBISDDSMPatchClassifierResNet(pl.LightningModule):
    def __init__(self, parameters):
        super(CBISDDSMPatchClassifierResNet, self).__init__()
        self.pretrained_model_name = 'resnet50'
        self.pretrained_model_path = parameters['pretrained_model_path']
        self.params_filename = parameters['filename']
        self.batch_size = parameters['batch_size']
        self.modification_type = parameters['modification_type']
        self.model_dest_folder = parameters['model_dst']
        self.num_classes = parameters['num_classes']
        self.loss_fn = nn.NLLLoss()

        # transfer learning parameters
        self.fc_n = -1
        self.basic_block_n = -1

        # store learning rates for different layers
        self.lr = dict()
        self.lr['init'] = parameters['lr_init']
        self.lr['fc'] = parameters['lr_fc']
        self.lr['conv'] = parameters['lr_conv']

        # lists to store outputs from each train/val step
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': []}
        self.test_history = {'test_acc': [], 'test_loss': []}

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

        custom_layers = get_layers(self.pretrained_model_name, self.num_classes, self.modification_type)
        self.model.fc = custom_layers['complete_fc']

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self, mode=None):

        # set parameters for conv blocks
        optim_params = []
        optim_params.append({'params': self.model.fc.parameters(), 'lr': self.lr['fc']})
        optim_params.append({'params': self.model.layer4.parameters(), 'lr': self.lr['conv']})
        optim_params.append({'params': self.model.layer3.parameters(), 'lr': self.lr['conv'] / 10})
        optim_params.append({'params': self.model.layer2.parameters(), 'lr': self.lr['conv'] / 100})
        optim_params.append({'params': self.model.layer1.parameters(), 'lr': self.lr['init']})
        optimizer = Adam(optim_params)

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Group {i + 1} - Learning Rate: {param_group['lr']}")

        return optimizer

    # freezes all layers in the model
    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # unfreeze last 'n' fully connected layers
    def unfreeze_last_n_fc_layers(self, n):

        # if n == -1 don't unfreeze any layers
        if n == -1:
            return 0

        n = n * 2  # since weights and bias are included as separate
        total_layers = len(list(self.model.fc.parameters()))

        # invalid n
        if n > total_layers:
            print(f"Warning: There are only {total_layers} layers in the model. Cannot unfreeze {n} layers.")

        # if n == 0 unfreeze all layers
        elif n == 0:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            for i, param in enumerate(self.model.fc.parameters()):
                if i >= (total_layers - n):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    # unfreeze last 'n' fully connected layers
    def unfreeze_last_n_conv_layers(self, n):

        # if n == -1 don't unfreeze any layers
        if n == -1:
            return 0

        model_layers = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1]

        total_layers = 0
        for i in range(n):
            total_layers += len(list(model_layers[i].parameters()))

        # invalid n
        if n > len(model_layers):
            print(
                f"Warning: There are only {len(model_layers)} layers in the model. Cannot unfreeze {n} layers.")

        # if n == 0 unfreeze all layers
        elif n == 0:
            for layer in model_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            for i in range(n):
                layer = model_layers[i]
                for param in layer.parameters():
                    param.requires_grad = True

    # set parameters for transfer learning
    def set_transfer_learning_params(self, unfreeze_n_fc, unfreeze_n_conv):
        self.fc_n = unfreeze_n_fc
        self.basic_block_n = unfreeze_n_conv
        self.freeze_all_layers()
        self.unfreeze_last_n_fc_layers(unfreeze_n_fc)
        self.unfreeze_last_n_conv_layers(unfreeze_n_conv)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.training_step_outputs.append((loss.item(), acc))
        return loss

    def on_train_epoch_end(self):
        num_items = len(self.training_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.training_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss / num_items
        avg_epoch_acc = cum_acc / num_items
        self.history['train_loss'].append(avg_epoch_loss)
        self.history['train_acc'].append(avg_epoch_acc)
        print(f'\nTraining Epoch({self.current_epoch}): loss: {avg_epoch_loss}, acc:{avg_epoch_acc}')
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.validation_step_outputs.append((loss.item(), acc))
        return loss

    def on_validation_epoch_end(self):
        num_items = len(self.validation_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.validation_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss / num_items
        avg_epoch_acc = cum_acc / num_items
        self.history['val_loss'].append(avg_epoch_loss)
        self.history['val_acc'].append(avg_epoch_acc)
        print(f'\nValidation Epoch({self.current_epoch}): loss: {avg_epoch_loss}, acc:{avg_epoch_acc}')
        self.validation_step_outputs.clear()
        self.log("val_loss", avg_epoch_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.test_step_outputs.append((loss.item(), acc))
        return loss

    def on_test_epoch_end(self):
        num_items = len(self.test_step_outputs)
        cum_loss = 0
        cum_acc = 0
        for loss, acc in self.test_step_outputs:
            cum_loss += loss
            cum_acc += acc

        avg_epoch_loss = cum_loss / num_items
        avg_epoch_acc = cum_acc / num_items
        print(f'Test Epoch loss: {avg_epoch_loss} Test epoch Acc: {avg_epoch_acc}')
        self.test_history['test_loss'].append(avg_epoch_loss)
        self.test_history['test_acc'].append(avg_epoch_acc)
        self.test_step_outputs.clear()

    def get_history(self):
        # remove the first validation epoch data
        self.history['val_loss'].pop(0)
        self.history['val_acc'].pop(0)
        return self.history

    def get_test_history(self):
        return self.test_history

    def clear_history(self):
        for key in self.history:
            self.history[key] = []

    def save_model(self, ckpt_path):
        # save the entire model
        model_architecture_path = os.path.join(ckpt_path, 'arc.pth')
        model_weights_path = os.path.join(ckpt_path, 'weights.pth')
        torch.save(self.model, model_architecture_path)
        torch.save(self.model.state_dict(), model_weights_path)
        print(f'Model saved at {ckpt_path}')
