import os
import sys
from model import MNISTClassifier
from train_model import TransferLearningPipiline

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


