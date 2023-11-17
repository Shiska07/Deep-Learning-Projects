import os
import argparse
import utils
from models import CBISDDSMPatchClassifier
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
    
            # initialize model
            custom_model = CBISDDSMPatchClassifier(parameters)

            # initialize transfer learning pipeline
            tl_pipeline = TransferLearningPipiline(custom_model, parameters)

            # train and test model
            tl_pipeline.train_model()
            tl_pipeline.test_model()

            # save model
            tl_pipeline.save_model()



if __name__ == "__main__":
    main()
    



    