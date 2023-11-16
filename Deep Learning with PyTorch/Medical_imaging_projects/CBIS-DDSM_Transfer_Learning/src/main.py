import os
import sys
import argparse
import utils
from models import CBISDDSMPatchClassifier
from train_model import TransferLearningPipiline
from attribution_maps import save_attribution_maps

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
    custom_model = CBISDDSMPatchClassifier(parameters)

    # initialize transfer learning pipeline
    tl_pipeline = TransferLearningPipiline(custom_model, parameters)

    # train model
    tl_pipeline.train_custom_fc_layers()

    #tl_pipeline.train()
    tl_pipeline.test_model()

    # save model
    model_arc_path, model_weights_path = tl_pipeline.save_model()

    save_attribution_maps(model_arc_path, model_weights_path, parameters['arrt_data_path'],
                          parameters['attribution_layer'], parameters['pretrained_model_name'])


if __name__ == "__main__":
    main()
    



    