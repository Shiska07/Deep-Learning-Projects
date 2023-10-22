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
        self.epochs_fc = parameters['epochs_fc']
        
        # number of total fully connected layers to unfreeze
        self.n_compfc = parameters['n_compfc']
        self.epochs_compfc = parameters['epochs_compfc']
        
        # no. of convolutional layers to unfreeze from last
        self.n_conv = parameters['n_conv']
        self.epochs_conv = parameters['epochs_conv']
        
        # no. of fully connected and convolutional layers to unfreeze for fine-tuning
        self.n_finetune_fc = parameters['n_finetune_fc']
        self.n_finetune_conv = parameters['n_finetune_conv']
        self.epochs_finetune = parameters['epochs_finetune']
        self.batch_size = parameters['batch_size']
        

    def train_custom_fc_layers(self):
        
        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_fc, -1)
        self.model.configure_optimizers('lr_fc')

        # train model
        trainer = pl.Trainer(max_epochs=self.epochs_fc, limit_train_batches=100, enable_checkpointing=False, logger=False)
        trainer.fit(self.model)

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
        trainer = pl.Trainer(max_epochs=self.epochs_compfc,
                             limit_train_batches=100, enable_checkpointing=False, logger=False)
        trainer.fit(self.model)

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
        trainer = pl.Trainer(max_epochs=self.epochs_conv,
                             limit_train_batches=100, enable_checkpointing=False, logger=False)
        trainer.fit(self.model)

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
        trainer = pl.Trainer(max_epochs=self.epochs_finetune,
                             limit_train_batches=100, enable_checkpointing=False, logger=False)
        trainer.fit(self.model)

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
        self.model.test()
                
    def get_model(self):
        return self.model
