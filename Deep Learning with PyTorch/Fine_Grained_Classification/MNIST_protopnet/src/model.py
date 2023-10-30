import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import pytorch_lightning as pl
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

class MNISTClassifier(pl.LightningModule):
    def __int__(self, parameters):
        super(MNISTClassifier, self).__init__()
        self.num_classes = parameters['num_classes']
        self.batch_size = parameters['batch_size']
        self.input_shape = parameters['input_shape']
        self.val_ratio = parameters['val_ratio']
        self.lr = parameters['lr']
        self.pretrained_model_path = parameters['pretrained_model_path']


        # lists to store outputs from each train/val step
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': []}


        # define model architecture
        self.loss_fn = nn.NLLLoss()
        self.conv_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxUnpool2d(kernel_size=2))
        self.conv_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxUnpool2d(kernel_size=2))
        self.flatten_1 = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.flatten_1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.LogSoftmax(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item()/self.batch_size
        self.training_step_outputs.append((loss.item(), acc))

        # print metrics every 100th batch
        if batch_idx % 100 == 0:
            print(f'\nTraining step[Epoch({self.current_epoch})|batch({batch_idx})]: loss: {loss.item()}, acc:{acc}')
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
        for params in self.model.parameters():
            print(params.requires_grad)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item()/self.batch_size
        self.validation_step_outputs.append((loss.item(), acc))
        print(f'\nVal step[batch({batch_idx})]: loss: {loss.item()}, acc:{acc}')
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
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.argmax(torch.exp(logits), 1)
        acc = (y_pred == y).sum().item() / self.batch_size
        self.test_step_outputs.append((loss.item(), acc))
        print(f'\nTest step: loss: {loss.item()}, acc:{acc}')
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

    def setup(self, stage=None):
        mnist_train = datasets.CIFAR10(
            root='./data', train=True, transform=transforms.ToTensor(), download=True)
        train_size = int((1 - self.val_size) * len(mnist_train))
        val_size = int(self.val_ratio * len(mnist_train))
        self.mnist_train, self.mnist_val = random_split(mnist_train, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        mnist_test = datasets.CIFAR10(
            root='./data', train=False, transform=transforms.ToTensor, download=True)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=1)