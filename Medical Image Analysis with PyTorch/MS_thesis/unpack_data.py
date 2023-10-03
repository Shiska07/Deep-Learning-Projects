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

# given .txt files containing image names and labels, creates an annotations .csv file


def create_labels_file(img_names_file, labels_file):

    annotations_path = '/content/drive/MyDrive/Colab Notebooks/Data/CUB_200_2011/image_labels.csv'

    # read class labels
    labels_df = pd.read_csv(
        labels_file, delim_whitespace=True, index_col=False, header=None)
    labels_ser = labels_df[1]

    # read image names
    img_df = pd.read_csv(img_names_file, delim_whitespace=True,
                         index_col=False, header=None)
    img_ser = img_df.iloc[:, 1]

    # create new df formatted as image_name, class_label
    df = pd.concat([img_ser, labels_ser], axis=1)
    df.to_csv(annotations_path, sep=',', index=False, header=False)

    return annotations_path
