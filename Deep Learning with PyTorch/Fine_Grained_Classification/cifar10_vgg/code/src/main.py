import os
import sys
import argparse
import utils
from model import CIFAR10Classifier
from train_model import TransferLearningPipiline

def main():
    
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(
        description='Load parameters from a JSON file.')

    # Add a command-line argument for the JSON file
    parser.add_argument('json_file', type=str,
                        help='Path to the JSON file containing parameters.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load parameters from the JSON file
    parameters = utils.load_parameters(args.json_file)
    
    # initialize model
    custom_model = CIFAR10Classifier(parameters)

    # initialize transfer learning pipeline
    tl_pipeline = TransferLearningPipiline(custom_model, parameters)

    # train model
    tl_pipeline.train_custom_fc_layers()
    #tl_pipeline.train()
    #tl_pipeline.test()


if __name__ == "__main__":
    
    with open('output_logs/console_outputs.txt', 'w+') as file:
        sys.stdout = file
        main()
        print('Console Outputs saved at "output_logs/console_outputs.txt"\n')

    sys.stdout = sys.__stdout__
    



    
