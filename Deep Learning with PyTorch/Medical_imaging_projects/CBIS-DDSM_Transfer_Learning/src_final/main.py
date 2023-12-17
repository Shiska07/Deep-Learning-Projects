import os
import argparse
import utils
import torch
import json
from train_model import TransferLearningPipiline
from custom_dataloaders import get_dataloaders
from models import CBISDDSMPatchClassifierVGG, CBISDDSMPatchClassifierResNet

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

            # get dataloaders
            dl_train, dl_val, dl_test = get_dataloaders(parameters['data_folder'], parameters['batch_size'],
                                                        parameters['validation_ratio'])
            # initialize model
            if 'vgg' in folder_name:
                custom_model = CBISDDSMPatchClassifierVGG(parameters)
            if 'resnet' in folder_name:
                custom_model = CBISDDSMPatchClassifierResNet(parameters)

            # initialize transfer learning pipeline
            tl_pipeline = TransferLearningPipiline(custom_model, parameters, dl_train, dl_val, dl_test)

            # train and test model
            final_history_path, best_model_path = tl_pipeline.train_model()
            test_history = tl_pipeline.test_model()

            # load model from checkpoint and save
            model_ckpt = os.path.join(best_model_path, 'best_model.ckpt')

            if 'vgg' in folder_name:
                final_model = CBISDDSMPatchClassifierVGG.load_from_checkpoint(model_ckpt, parameters=parameters)
                test_history = tl_pipeline.test_model()
                final_model.save_model(best_model_path)
            if 'resnet' in folder_name:
                final_model = CBISDDSMPatchClassifierResNet.load_from_checkpoint(model_ckpt, parameters=parameters)
                test_history = tl_pipeline.test_model()
                final_model.save_model(best_model_path)

            test_result_path = os.path.join(final_history_path, 'test_results.json')
            with open(test_result_path, 'w') as json_file:
                json.dump(test_history, json_file)



if __name__ == "__main__":
    main()
    



    