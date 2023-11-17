import os
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import pytorch_lightning as pl
from custom_layers import get_layers
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

class CBISDDSMPatchClassifier(pl.LightningModule):
    def __init__(self, parameters):
        super(CBISDDSMPatchClassifier, self).__init__()
        self.pretrained_model_name = parameters['pretrained_model_name']
        self.pretrained_model_path = parameters['pretrained_model_path']
        self.batch_size = parameters['batch_size']
        self.resizing_factor = parameters['resizing_factor']
        self.modification_type = parameters['modification_type']
        self.data_folder = parameters['data_folder']
        self.model_dest_folder = parameters['model_dst']
        self.num_classes = parameters['num_classes']
        self.val_ratio = parameters['validation_ratio']
        self.loss_fn = nn.NLLLoss()
        self.ddsm_train, self.ddsm_val, self.ddsm_test = None, None, None

        # transfer learning parameters
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.classifiers_n = -1
        self.features_n = -1

        # activation maps parameters
        self.layers_for_gradcam = [29]
        self.activation_maps_enabled = False
        
        # store learning rates for different layers
        self.lr = dict()
        self.lr['fc'] = parameters['lr_fc']
        self.lr['compfc'] = parameters['lr_compfc']
        self.lr['conv'] = parameters['lr_conv']
        self.lr['finetune'] = parameters['lr_finetune']
        
        # lists to store outputs from each train/val step
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.history = {'train_loss': [], 'train_acc': [],
                    'val_loss': [], 'val_acc': []}

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


        # layer corresponding to modification type
        if self.pretrained_model_name in ['vgg16', 'vgg19']:
            custom_layers = get_layers(self.pretrained_model_name, self.num_classes, self.modification_type)

            if self.modification_type == 'final_fc':
                self.model.classifier[-1] = custom_layers['final_classifier']

            else:
                self.model.features[-1] = custom_layers['final_features']
                self.model.classifier = custom_layers['complete_classifier']

        elif self.pretrained_model_name in ['resnet34', 'resnet50']:
            custom_layers = get_layers(self.pretrained_model_name, self.num_classes, self.modification_type)
            self.model.fc = custom_layers['complete_fc']

        elif self.pretrained_model_name in ['efficientnet_b6', 'efficientnet_b7']:
            custom_layers = get_layers(self.pretrained_model_name, self.num_classes, self.modification_type)
            self.model.classifier = custom_layers['complete_fc']


    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self, mode = None):
        if mode is None:
            optimizer = Adam(filter(lambda p: p.requires_grad,
                            self.model.parameters()), lr=0.001)
        else:
            optimizer = Adam(filter(lambda p: p.requires_grad,
                                self.model.parameters()), lr=self.lr[mode])
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

        n = n*2  # since weights and bias are included as separate
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
        print(f'\nTraining Epoch({self.current_epoch}): loss: {avg_epoch_loss}, acc:{avg_epoch_acc}')
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
        print(f'\nValidation Epoch({self.current_epoch}): loss: {avg_epoch_loss}, acc:{avg_epoch_acc}')
        self.validation_step_outputs.clear()

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
        self.test_step_outputs.clear()

    def setup(self, stage = None):
        if stage == 'fit' or stage == 'validate':
            transform = transforms.Compose([
                transforms.Resize((self.resizing_factor, self.resizing_factor)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

            # load train and validation datasets
            ddsm_train = datasets.ImageFolder(root=str(self.data_folder + 'train'),
                                              transform=transform)
            val_size = int(self.val_ratio * len(ddsm_train))
            train_size = len(ddsm_train) - val_size
            self.ddsm_train, self.ddsm_val = random_split(ddsm_train, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.ddsm_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ddsm_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.resizing_factor, self.resizing_factor)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # load test dataset
        self.ddsm_test = datasets.ImageFolder(root=str(self.data_folder + 'test'),
                                              transform=transform)
        return DataLoader(self.ddsm_test, batch_size=self.batch_size, num_workers=8)

    def get_history(self):
        # remove the first validation epoch data
        self.history['val_loss'].pop(0)
        self.history['val_acc'].pop(0)
        return self.history

    def clear_history(self):
        for key in self.history:
            self.history[key] = []
    
    def save_model(self):
        # save the entire model
        model_architecture_path = os.path.join(self.model_dest_folder, str(self.pretrained_model_name) + '_'+ str(self.batch_size) + 'arc.pth')
        model_weights_path = os.path.join(self.model_dest_folder, str(self.pretrained_model_name) + '_' + str(
            self.batch_size) + 'weights.pth')
        torch.save(self.model, model_architecture_path )
        torch.save(self.model.state_dict(), model_architecture_path )
        print(f'Model saved at {self.model_dest_folder}')

