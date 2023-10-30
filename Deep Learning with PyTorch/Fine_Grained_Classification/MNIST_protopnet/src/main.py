import os
import sys
from model import MNISTClassifier
import pytorch_lightning as pl

def main():

    # define parameters
    parameters = dict()
    parameters['num_classes'] = 10
    parameters['batch_size'] = 32
    parameters['lr'] = 0.001
    parameters['input_shape'] = (28, 28)
    parameters['pretrained_model_path'] = None
    parameters['val_ratio'] = 0.3
    # initalize model
    custom_model = MNISTClassifier(parameters)

    # intitalize model
    epochs = 20
    trainer = pl.Trainer(max_epochs=epochs,enable_checkpointing=False, logger=False)

    # fit model
    trainer.fit(custom_model)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


