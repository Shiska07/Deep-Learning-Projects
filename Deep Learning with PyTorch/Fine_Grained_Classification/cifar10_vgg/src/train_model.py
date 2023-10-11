import os
import utils
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

class TransferLearningPipiline:
    def __init__(self, model, parameters):
        self.model = model
        self.pretrained_model_name = parameters['pretrained_model_name']
        self.log_path = parameters['logs_path']
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

        # initialize log path
        log_dir = self.log_path +'/' + str(self.pretrained_model_name) + '/batchsz' + str(self.batch_size) + '/' + 'fc/'

        # create directories if non-existent
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {log_dir}: {e}")

        # initalize logger
        logger = TensorBoardLogger(log_dir)
        logger.log_hyperparams({'epochs': self.epochs_fc,
                                'batch_size': self.batch_size,
                                'name': self.pretrained_model_name})

        # train model
        trainer = pl.Trainer(max_epochs=self.epochs_fc, logger=logger)
        trainer.fit(self.model)
        print(f'Training Complete. Results logges at {log_dir}')

        # get training history
        history = self.model.get_history()

        # plot history
        plots_path = utils.save_plots(history, self.plots_dir, self.pretrained_model_name, self.batch_size, 'fc')
        printf(f'Custom fc training complete. Plots saved at {plots_path}. Logs saved at {log_dir}.\n')
        
        
    def train_all_fc_layers(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_compfc, -1)

        # initialize log path
        log_dir = self.log_path + '/' +\
            str(self.pretrained_model_name) + '/batchsz' + \
            str(self.batch_size) + '/' + 'compfc/'

        # create directories if non-existent
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {log_dir}: {e}")

        # initalize logger
        logger = TensorBoardLogger(log_dir)
        logger.log_hyperparams({'epochs': self.epochs_compfc,
                                'batch_size': self.batch_size,
                                'name': self.pretrained_model_name})

        # train model
        trainer = pl.Trainer(max_epochs=self.epochs_compfc, logger=logger)
        trainer.fit(self.model)
        print(f'Training Complete. Results logges at {log_dir}')

        # get training history
        history = self.model.get_history()

        # plot history
        plots_path = utils.save_plots(history, self.plots_dir,
                         self.pretrained_model_name, self.batch_size, 'compfc')

        printf(f'Complete fc training complete. Plots saved at {plots_path}. Logs saved at {log_dir}.\n')


    def train_conv_layers(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(-1, self.n_conv)

        # initialize log path
        log_dir = self.log_path + '/' +\
            str(self.pretrained_model_name) + '/batchsz' + \
            str(self.batch_size) + '/' + 'conv/'

        # create directories if non-existent
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {log_dir}: {e}")

        # initalize logger
        logger = TensorBoardLogger(log_dir)
        logger.log_hyperparams({'epochs': self.epochs_conv,
                                'batch_size': self.batch_size,
                                'name': self.pretrained_model_name})

        # train model
        trainer = pl.Trainer(max_epochs=self.epochs_conv, logger=logger)
        trainer.fit(self.model)
        print(f'Training Complete. Results logges at {log_dir}')

        # get training history
        history = self.model.get_history()

        # save history
        plots_path = utils.save_plots(history, self.plots_dir,
                         self.pretrained_model_name, self.batch_size, 'conv')

        printf(f'Convolutional layers training complete. Plots saved at {plots_path}. Logs saved at {log_dir}.\n')


    def fine_tune_models(self):

        # freeze all layers except the last two fc layers
        self.model.set_transfer_learning_params(self.n_finetune_fc, self.n_finetune_conv)

        # initialize log path
        log_dir = self.log_path + '/' +\
            str(self.pretrained_model_name) + '/batchsz' + \
            str(self.batch_size) + '/' + 'finetune/'

        # create directories if non-existent
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {log_dir}: {e}")

        # initalize logger
        logger = TensorBoardLogger(log_dir)
        logger.log_hyperparams({'epochs': self.epochs_finetune,
                                'batch_size': self.batch_size,
                                'name': self.pretrained_model_name})

        # train model
        trainer = pl.Trainer(max_epochs=self.epochs_finetune, logger=logger)
        trainer.fit(self.model)
        print(f'Training Complete. Results logges at {log_dir}')

        # get training history
        history = self.model.get_history()

        # plot history
        plots_path = utils.save_plots(history, self.plots_dir,
                         self.pretrained_model_name, self.batch_size, 'finetune')

        printf(f'Model fine-tuning complete. Plots saved at {plots_path}. Logs saved at {log_dir}.\n')


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
