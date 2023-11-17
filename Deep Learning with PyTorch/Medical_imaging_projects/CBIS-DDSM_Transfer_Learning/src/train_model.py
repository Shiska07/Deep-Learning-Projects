import os
import utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class TransferLearningPipiline:
    def __init__(self, model, parameters):
        self.model = model
        self.params = parameters
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
        checkpoint_path = os.path.join(self.model_dest, self.pretrained_model_name, self.batch_size, mode, self.model.lr[mode])
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            dirpath=checkpoint_path,
            every_n_epochs = 10,
            filename="cbis-ddsm-{epoch:02d}-{val_loss:.2f}",
        )
        self.trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=self.epochs[mode],
                                 enable_progress_bar=False, enable_checkpointing=True, callbacks=[checkpoint_callback], logger=False)

    def train_custom_fc_layers(self):
        
        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_fc, -1)
        self.model.configure_optimizers('lr_fc')
        self.model.clear_history()

        # train model
        self.initalize_trainer('fc')
        self.trainer.fit(self.model)

        # save history plot and data
        history = self.model.get_history()
        keys = ['epochs_fc', 'lr_fc']
        h_params = {key: self.params[key] for key in keys}
        plots_path = utils.save_plots(history, self.plots_path, self.pretrained_model_name, self.batch_size, 'fc', self.model.lr['lr_fc'])
        print(f'Custom fc training complete. Plots saved at {plots_path}\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.batch_size, 'fc', self.model.lr['lr_fc'], h_params)


        print(f'Successfully saved custom fc layers history file at {history_path}.\n')


        
    def train_all_fc_layers(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_compfc, -1)
        self.model.configure_optimizers('lr_compfc')
        self.model.clear_history()

        # train model
        self.initalize_trainer('compfc')
        self.trainer.fit(self.model)

        # save history plot and data
        history = self.model.get_history()
        keys = ['epochs_compfc', 'lr_compfc']
        h_params = {key: self.params[key] for key in keys}
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.batch_size, 'compfc', self.model.lr['lr_compfc'])
        print(f'Complete fc training complete. Plots saved at {plots_path}\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.batch_size, 'compfc', self.model.lr['lr_compfc'], h_params)


        print(f'Successfully saved complete fc layers history file at {history_path}.\n')


    def train_conv_layers(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(-1, self.n_conv)
        self.model.configure_optimizers('lr_conv')
        self.model.clear_history()

        # train model
        self.initalize_trainer('conv')
        self.trainer.fit(self.model)

        # save history plot and data
        history = self.model.get_history()
        keys = ['epochs_conv', 'lr_conv']
        h_params = {key: self.params[key] for key in keys}
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.batch_size, 'conv', self.model.lr['lr_conv'])
        print(f'Convolutional layers training complete. Plots saved at {plots_path}\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.batch_size, 'conv', self.model.lr['lr_conv'], h_params)

        print(f'Successfully saved convolutional layers history file at {history_path}.\n')


    def fine_tune_models(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_finetune_fc, self.n_finetune_conv)
        self.model.configure_optimizers('lr_finetune')
        self.model.clear_history()

        # train model
        self.initalize_trainer('finetune')
        self.trainer.fit(self.model)

        # save history plot and data
        history = self.model.get_history()
        keys = ['epochs_finetune', 'lr_finetune']
        h_params = {key: self.params[key] for key in keys}
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.batch_size, 'finetune', self.model.lr['lr_finetune'])
        print(f'Model fine-tuning complete. Plots saved at {plots_path}.\n')

        history_path = utils.save_history(history, self.epoch_history_path,
                                          self.pretrained_model_name, self.batch_size, 'finetune', self.model.lr['lr_finetune'], h_params)
        print(f'Successfully saved fine-tuning history file at {history_path}.\n')


    # complete transfer learning pipeline
    def train_model(self):
        self.train_custom_fc_layers()
        self.train_all_fc_layers()
        self.train_conv_layers()
        self.fine_tune_models()
    
    # run test    
    def test_model(self):
        # call test off the most current trainer
        self.trainer.test(self.model)
                
    def get_model(self):
        return self.model
    
    def save_model(self):
        self.model.save_model()
