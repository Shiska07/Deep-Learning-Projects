import os
import argparse
import utils
import torch
import json
from models import CBISDDSMPatchClassifierVGG, CBISDDSMPatchClassifierResNet
from train_model import TransferLearningPipiline

def main():

    parser = argparse.ArgumentParser(description="Load JSON files from a directory")
    parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    args = parser.parse_args()
    folder_name = args.folder_name

    directory_path = os.path.abspath(folder_name)


    for filename in os.listdir(directory_path):

        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)

            parameters = utils.load_parameters(file_path)
            parameters['filename'] = filename

            custom_model = None

            # initialize model
            if 'vgg' in folder_name:
                custom_model = CBISDDSMPatchClassifierVGG(parameters)
            if 'resnet' in folder_name:
                custom_model = CBISDDSMPatchClassifierResNet(parameters)

            # initialize transfer learning pipeline
            tl_pipeline = TransferLearningPipiline(custom_model, parameters)

            # train and test model
            final_history_path = tl_pipeline.train_model()
            test_history = tl_pipeline.test_model()

            test_result_path = os.path.join(final_history_path, 'test_results.json')
            with open(test_result_path, 'w') as json_file:
                json.dump(test_history, json_file)


if __name__ == "__main__":
    main()
    



    