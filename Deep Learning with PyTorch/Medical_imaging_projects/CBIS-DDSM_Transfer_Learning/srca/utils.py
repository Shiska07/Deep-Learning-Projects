# dataset object for testing so that image name is retained
import os
import json
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset


# these are the mean and std of the data per channel
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denormalize(tensor):
    tensor = tensor*std + mean
    return tensor

# function for viewling image
def show_img(img):
    # arrange channels
    img = img.numpy().transpose((1,2,0))

    # use mean and std values
    img = denormalize(img)

    # clip values and view image
    rgb_img = np.clip(img,0,1)

    return np.float32(rgb_img)


def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)

    normalized_img = (img - min_val) / (max_val - min_val)
    normalized_img *= 255
    return normalized_img.astype(int)


def load_parameters(json_file):
    try:
        with open(json_file, 'r') as file:
            parameters = json.load(file)
        return parameters
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON file '{json_file}' is not a valid JSON file.")
        return None


class CustomTestDataset(Dataset):
    def __init__(self, img_dir,annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_name