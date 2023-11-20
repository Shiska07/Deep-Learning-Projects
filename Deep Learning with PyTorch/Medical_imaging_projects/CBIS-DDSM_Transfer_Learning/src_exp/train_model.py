import os
import utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class TransferLearningPipiline:
    def __init__(self, model, parameters):
        self.model = model
        self.params = parameters
        self.params_filename = parameters['filename']
        self.pretrained_model_name = parameters['pretrained_model_name']
        self.plots_path = parameters['plots_path']
        self.epoch_history_path = parameters['epoch_history_path']
        self.model_dest = parameters['model_dst']
        
        # funny connected layers to unfreeze from last
        self.n_fc = parameters['n_fc']                                   
        
        # number of total fully connected layers to unfreeze
        self.n_compfc = parameters['n_compfc']
        
        # no. of convolutional layers to unfreeze from last
        self.n_conv = parameters['n_conv']
        
        # no. of fully connected and convolutional layers to unfreeze for fine-tuning
        self.n_finetune_fc = parameters['n_finetune_fc']
        self.n_finetune_conv = parameters['n_finetune_conv']
        self.batch_size = parameters['batch_size']
        
        self.epochs = dict()
        self.epochs['fc'] = parameters['epochs_fc']
        self.epochs['compfc'] = parameters['epochs_compfc']
        self.epochs['conv'] = parameters['epochs_conv']
        self.epochs['finetune'] = parameters['epochs_finetune']

        self.trainer = None

    
    def initalize_trainer(self, mode):
        # checkpoint for each part of the training
        self.checkpoint_path = os.path.join(self.model_dest, self.pretrained_model_name, str(self.params_filename))

        try:
            os.makedirs(self.checkpoint_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {self.checkpoint_path}: {e}")

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # You can change this to the metric you want to monitor
            dirpath=self.checkpoint_path,  # Directory to save checkpoints
            filename=f'best_model_{mode}',  # Prefix for the checkpoint filenames
            save_top_k=1,  # Save the best model only
            mode='min'  # 'min' means to save the checkpoint with the minimum value of the monitored quantity
        )
        
        self.trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=self.epochs[mode], callbacks=[early_stopping_callback, checkpoint_callback],
                                 enable_progress_bar=False, check_val_every_n_epoch=1, enable_checkpointing=True, default_root_dir=self.checkpoint_path, logger=True)

        '''
        self.trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=self.epochs[mode],
                                  enable_progress_bar=False, enable_checkpointing=True,
                                  default_root_dir=self.checkpoint_path, logger=False)
        '''

    def train_custom_fc_layers(self):
        
        # only update fc optimizers at this stage
        config_fc = 1
        config_conv = 0
        self.model.set_transfer_learning_params(self.n_fc, -1, config_fc, config_conv)
        self.model.clear_history()

        # train model
        self.initalize_trainer('fc')
        print('BEGIN training custom fully connected layers.\n')
        self.trainer.fit(self.model)
        print('END training custom fully connected layers.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = utils.save_plots(history, self.plots_path, self.pretrained_model_name, self.params_filename, self.batch_size, 'fc')
        print(f'Custom fc training complete. Plots saved at {plots_path}\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.params_filename, self.batch_size, 'fc', self.params)

        print(f'Successfully saved custom fc layers history file at {history_path}.\n')


        
    def train_all_fc_layers(self):

        # no updates to optimizer state at this stage
        config_fc = 0
        config_conv = 0
        self.model.set_transfer_learning_params(self.n_compfc, -1, config_fc, config_conv)
        self.model.clear_history()

        # train model
        self.initalize_trainer('compfc')
        print('BEGIN training all fully connected layers.\n')
        self.trainer.fit(self.model)
        print('END training all fully connected layers.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.params_filename, self.batch_size, 'compfc')
        print(f'Complete fc training complete. Plots saved at {plots_path}\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.params_filename, self.batch_size, 'compfc', self.params)

        print(f'Successfully saved complete fc layers history file at {history_path}.\n')


    def train_conv_layers(self):

        # only update optimizers for conv layer during this step
        config_fc = 0
        config_conv = 1
        self.model.set_transfer_learning_params(-1, self.n_conv, config_fc, config_conv)
        self.model.configure_optimizers('conv')
        self.model.clear_history()

        # train model
        self.initalize_trainer('conv')
        print('BEGIN training conv layers.\n')
        self.trainer.fit(self.model)
        print('BEGIN training conv connected layers.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.params_filename,self.batch_size, 'conv')
        print(f'Convolutional layers training complete. Plots saved at {plots_path}\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.params_filename,self.batch_size, 'conv', self.params)

        print(f'Successfully saved convolutional layers history file at {history_path}.\n')


    def fine_tune_models(self):

        # optimizers update for this state
        config_fc = 0
        config_conv = 0
        self.model.set_transfer_learning_params(self.n_finetune_fc, self.n_finetune_conv, config_fc, config_conv)
        self.model.clear_history()

        # train model
        self.initalize_trainer('finetune')
        print('BEGIN fine-tuning model.\n')
        self.trainer.fit(self.model)
        print('END fine-tuning model.\n')

        # save history plot and data
        history = self.model.get_history()
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.params_filename, self.batch_size, 'finetune')
        print(f'Model fine-tuning complete. Plots saved at {plots_path}.\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.params_filename, self.batch_size, 'finetune', self.params)
        print(f'Successfully saved fine-tuning history file at {history_path}.\n')

        return history_path


    # complete transfer learning pipeline
    def train_model(self):
        if self.pretrained_model_name in ['vgg16', 'vgg19']:
            self.train_custom_fc_layers()
            self.train_all_fc_layers()
            self.train_conv_layers()
            final_history_path = self.fine_tune_models()
            return final_history_path, self.checkpoint_path

        elif self.pretrained_model_name in ['resnet34', 'resnet50']:
            self.train_custom_fc_layers()
            self.train_conv_layers()
            final_history_path = self.fine_tune_models()
            return final_history_path, self.checkpoint_path
    
    # run test    
    def test_model(self):
        # call test off the most current trainer
        self.trainer.test(self.model)
        return self.model.get_test_history()
                
    def get_model(self):
        return self.model

    def save_model(self):
        self.model.save_model(self.checkpoint_path)

