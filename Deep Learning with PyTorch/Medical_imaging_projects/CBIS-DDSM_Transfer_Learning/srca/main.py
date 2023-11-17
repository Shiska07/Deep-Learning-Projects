import os
import argparse
from utils import load_parameters
from attribution_maps import save_attribution_maps

def main():

    parser = argparse.ArgumentParser(description="Load JSON files from a directory")
    parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    args = parser.parse_args()
    folder_name = args.folder_name
    directory_path = os.path.abspath(folder_name)

    for filename in os.listdir(directory_path):

        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)

            parameters = load_parameters(file_path)
    
            # save model
            model_arc_path = os.path.join(parameters['model_dst'], parameters['pretrained_model_name'] + '_'+ parameters['batch_size'] + 'arc.pth')
            model_weights_path = os.path.join(parameters['model_dst'], parameters['pretrained_model_name'] + '_'+ parameters['batch_size'] + 'weights.pth')

            save_attribution_maps(model_arc_path, model_weights_path, parameters['arrt_data_path'],
                                 parameters['attribution_layer'], parameters['pretrained_model_name'])


if __name__ == "__main__":
    main()
    
