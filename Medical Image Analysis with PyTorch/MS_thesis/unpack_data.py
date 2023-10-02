'''
This script is used to unpack the CUB_200_2011.tgz file and create folder with all images and the annotation files.
'''
import os
import shutil
import tarfile
import numpy as np
import pandas as pd


def CUB_200_2011_unpack(tgz_file_path):
    with tarfile.open(tgz_file_path, 'r:gz') as tar:
        tar.extractall()
        
# returns path to directory with all images
def extract_all_images_folder(img_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(('jpg', '.jpeg', '.png')):
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(out_dir, file)
                shutil.copy2(src_file_path, dst_file_path)
    print('Copying Complete')
    return out_dir


# given .txt files containing image names and labels, creates an annotations .csv file
def create_labels_file():

    annotations_path = 'CUB_200_2011/image_labels.csv'
    data_dir = 'CUB_200_2011'

    img_dir = 'CUB_200_2011/images'
    img_file = 'CUB_200_2011/images.txt'
    labels_file = 'CUB_200_2011/image_class_labels.txt'
    
    # read class labels
    labels_df = pd.read_csv(labels_file, delim_whitespace=True, index_col=False, header=None)
    labels_ser = labels_df[1]

    # read image names
    img_df = pd.read_csv(img_file, sep='/', index_col=False, header=None)
    img_ser = img_df[1]

    # create new df formatted as image_name, class_label
    df = pd.concat([img_ser, labels_ser], axis=1)
    df.to_csv(annotations_path, index=False, header=False)

    return annotations_path
