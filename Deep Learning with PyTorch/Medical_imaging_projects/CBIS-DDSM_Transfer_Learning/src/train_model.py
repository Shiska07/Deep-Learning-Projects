import os
import utils
import pytorch_lightning as pl

class TransferLearningPipiline:
    def __init__(self, model, parameters):
        self.model = model
        self.pretrained_model_name = parameters['pretrained_model_name']
        self.plots_path = parameters['plots_path']
        
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

    
    def initalize_trainer(self, mode):
        self.trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=self.epochs[mode],
                                  limit_val_batches=50, enable_progress_bar=False, limit_test_batches=50, enable_checkpointing=False, logger=False)

    def train_custom_fc_layers(self):
        
        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_fc, -1)
        self.model.configure_optimizers('lr_fc')

        # train model
        self.initalize_trainer('fc')
        self.trainer.fit(self.model)

        # get training history
        history = self.model.get_history()

        # plot history
        plots_path = utils.save_plots(history, self.plots_path, self.pretrained_model_name, self.batch_size, 'fc')
        print(f'Custom fc training complete. Plots saved at {plots_path}\n')
        
        
    def train_all_fc_layers(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_compfc, -1)
        self.model.configure_optimizers('lr_compfc')

        # train model
        self.initalize_trainer('compfc')
        self.trainer.fit(self.model)

        # get training history
        history = self.model.get_history()

        # plot history
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.batch_size, 'compfc')

        print(f'Complete fc training complete. Plots saved at {plots_path}\n')


    def train_conv_layers(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(-1, self.n_conv)
        self.model.configure_optimizers('lr_conv')

        # train model
        self.initalize_trainer('conv')
        self.trainer.fit(self.model)

        # get training history
        history = self.model.get_history()

        # save history
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.batch_size, 'conv')

        print(f'Convolutional layers training complete. Plots saved at {plots_path}\n')


    def fine_tune_models(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_finetune_fc, self.n_finetune_conv)
        self.model.configure_optimizers('lr_finetune')

        # train model
        self.initalize_trainer('finetune')
        self.trainer.fit(self.model)

        # get training history
        history = self.model.get_history()

        # plot history
        plots_path = utils.save_plots(history, self.plots_path,
                         self.pretrained_model_name, self.batch_size, 'finetune')

        print(f'Model fine-tuning complete. Plots saved at {plots_path}.\n')


    # complete transfer learning pipeline
    def train(self):
        self.train_custom_fc_layers()
        self.train_all_fc_layers()
        self.train_conv_layers()
        self.fine_tune_models()
    
    # run test    
    def test(self):
        # call test off the most current trainer
        self.trainer.test()
                
    def get_model(self):
        return self.model
    
    def save_model(self):
        self.model.save_model()