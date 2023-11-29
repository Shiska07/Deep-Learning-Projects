import os
import pytorch_lightning as pl
from utils import save_plots, save_history
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models import CBISDDSMPatchClassifierVGG, CBISDDSMPatchClassifierResNet


class TransferLearningPipiline:
    def __init__(self, model, parameters, train_dataloader, val_dataloader, test_dataloader):
        self.model = model
        self.params = parameters
        self.params_filename = parameters['filename']
        self.pretrained_model_name = parameters['pretrained_model_name']
        self.plots_path = parameters['plots_path']
        self.epoch_history_path = parameters['epoch_history_path']
        self.model_dest = parameters['model_dst']

        # save dataloaders
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

        # funny connected layers to unfreeze from last
        self.n_fc = parameters['n_fc']

        # no. of convolutional layers to unfreeze from last
        self.n_conv = parameters['n_conv']

        # no. of fully connected and convolutional layers to unfreeze for fine-tuning
        self.n_finetune_fc = parameters['n_finetune_fc']
        self.n_finetune_conv = parameters['n_finetune_conv']
        self.batch_size = parameters['batch_size']

        self.epochs = dict()
        self.epochs['fc'] = parameters['epochs_fc']
        self.epochs['conv'] = parameters['epochs_conv']
        self.epochs['finetune'] = parameters['epochs_finetune']

        self.trainer = None

    def load_model_from_checkpoint(self, mode):
        model_path = os.path.join(self.checkpoint_path, 'best_model.ckpt')
        if 'vgg' in self.pretrained_model_name:
            self.model = CBISDDSMPatchClassifierVGG.load_from_checkpoint(model_path, parameters=self.params)
            print(f'Model for mode {mode} loaded from {self.checkpoint_path}')
        elif 'resnet' in self.pretrained_model_name:
            self.model = CBISDDSMPatchClassifierResNet.load_from_checkpoint(model_path,
                                                                            parameters=self.params)
            print(f'Model for mode {mode} loaded from {self.checkpoint_path}')

    def initalize_trainer(self, mode):
        # checkpoint for each part of the training
        self.checkpoint_path = os.path.join(self.model_dest, self.pretrained_model_name, str(self.params_filename),
                                            mode, 'checkpoints')
        self.logs_path = os.path.join(self.model_dest, self.pretrained_model_name, str(self.params_filename), 'logs')

        try:
            os.makedirs(self.checkpoint_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {self.checkpoint_path}: {e}")
        try:
            os.makedirs(self.logs_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {self.checkpoint_path}: {e}")

        if mode == 'fc':
            pat = 7
        elif mode == 'conv':
            pat = 5
        else:
            pat = 5

        early_stopping_callback = EarlyStopping(
            monitor='val_loss', min_delta=0.01,
            patience=pat,
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.checkpoint_path,  # Directory to save checkpoints
            filename=f'best_model',  # Prefix for the checkpoint filenames
            save_top_k=1,  # Save the best model only
            mode='min',
            every_n_epochs=1
        )

        self.trainer = pl.Trainer(accelerator="gpu", devices="auto", max_epochs=self.epochs[mode],
                                  callbacks=[early_stopping_callback, checkpoint_callback],
                                  enable_progress_bar=False, check_val_every_n_epoch=1, enable_checkpointing=True,
                                  default_root_dir=self.logs_path, logger=True)

    def train_custom_fc_layers(self):

        self.model.set_transfer_learning_params(self.n_fc, -1)
        self.model.clear_history()

        # train model
        self.initalize_trainer('fc')
        print('BEGIN training custom fully connected layers.\n')
        self.trainer.fit(self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)
        print('END training custom fully connected layers.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = save_plots(history, self.plots_path, self.pretrained_model_name, self.params_filename,
                                self.batch_size, 'fc')
        print(f'Custom fc training complete. Plots saved at {plots_path}\n')

        history_path = save_history(history, self.epoch_history_path,
                                    self.pretrained_model_name, self.params_filename, self.batch_size, 'fc',
                                    self.params)

        print(f'Successfully saved custom fc layers history file at {history_path}.\n')

    def train_conv_layers1(self):

        self.load_model_from_checkpoint('conv1')
        self.model.set_transfer_learning_params(-1, self.n_conv)
        self.model.clear_history()

        # train model
        self.initalize_trainer('conv')
        print('BEGIN training final conv layers.\n')
        self.trainer.fit(self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)
        print('BEGIN training final conv connected layers.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = save_plots(history, self.plots_path,
                                self.pretrained_model_name, self.params_filename, self.batch_size, 'conv1')
        print(f'Convolutional layers training complete. Plots saved at {plots_path}\n')

        history_path = save_history(history, self.epoch_history_path,
                                    self.pretrained_model_name, self.params_filename, self.batch_size, 'conv1',
                                    self.params)

        print(f'Successfully saved convolutional layers history file at {history_path}.\n')

    def train_conv_layers2(self):

        self.load_model_from_checkpoint('conv2')
        if 'vgg' in self.pretrained_model_name:
            self.model.set_transfer_learning_params(-1, self.n_conv * 2)
        else:
            self.model.set_transfer_learning_params(-1, self.n_conv + 1)
        self.model.clear_history()

        # train model
        self.initalize_trainer('conv')
        print('BEGIN training all conv layers.\n')
        self.trainer.fit(self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)
        print('END training all conv layers.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = save_plots(history, self.plots_path,
                                self.pretrained_model_name, self.params_filename, self.batch_size, 'conv2')
        print(f'Convolutional layers training complete. Plots saved at {plots_path}\n')

        history_path = save_history(history, self.epoch_history_path,
                                    self.pretrained_model_name, self.params_filename, self.batch_size, 'conv2',
                                    self.params)

        print(f'Successfully saved convolutional layers history file at {history_path}.\n')

    def fine_tune_models(self):

        self.load_model_from_checkpoint('finetune')
        self.model.set_transfer_learning_params(self.n_finetune_fc, self.n_finetune_conv)
        self.model.clear_history()

        # train model
        self.initalize_trainer('finetune')
        print('BEGIN fine-tuning model.\n')
        self.trainer.fit(self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)
        print('END fine-tuning model.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = save_plots(history, self.plots_path,
                                self.pretrained_model_name, self.params_filename, self.batch_size, 'finetune')
        print(f'Model fine-tuning complete. Plots saved at {plots_path}.\n')

        history_path = save_history(history, self.epoch_history_path,
                                    self.pretrained_model_name, self.params_filename, self.batch_size, 'finetune',
                                    self.params)
        print(f'Successfully saved fine-tuning history file at {history_path}.\n')

        return history_path

    # complete transfer learning pipeline
    def train_model(self):
        if self.pretrained_model_name == 'vgg19':
            self.train_custom_fc_layers()
            # self.train_all_fc_layers()
            self.train_conv_layers1()
            self.train_conv_layers2()
            final_history_path = self.fine_tune_models()
            return final_history_path, self.checkpoint_path

        elif self.pretrained_model_name == 'resnet50':
            self.train_custom_fc_layers()
            self.train_conv_layers1()
            self.train_conv_layers2()
            final_history_path = self.fine_tune_models()
            return final_history_path, self.checkpoint_path

    # run test
    def test_model(self):
        # call test off the most current trainer
        self.trainer.test(self.model, self.test_dl)
        return self.model.get_test_history()

    def get_model(self):
        return self.model

    def save_model(self):
        self.model.save_model(self.checkpoint_path)

