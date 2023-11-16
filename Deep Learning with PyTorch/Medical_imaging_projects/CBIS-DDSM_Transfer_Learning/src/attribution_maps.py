import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split
from attribution_model import TrainedModel

import cv2
import captum
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# these are the mean and std of the data per channel
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])

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


def get_attention_map(model, x, x_fullsize, layer, alpha):
    model.to(device)
    model.eval()

    # convert data to tensor dimension
    input_img = x.unsqueeze(0)
    logps = model(input_img)
    output = torch.exp(logps)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()

    # get grad cam results
    layer_gradcam = LayerGradCam(model, model.model.features[layer])
    attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

    upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, x_fullsize.shape[1:])

    # overlay heatmap over rgb image
    rgb_fullsz_image = x_fullsize.cpu().permute(1,2,0).detach().numpy()
    rgb_fullsz_image = normalize_image(rgb_fullsz_image)
    heatmap_image = normalize_image(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy())

    # Resize heatmap if sizes are different
    if rgb_fullsz_image.shape[:2] != heatmap_image.shape[:2]:
        heatmap_image = cv2.resize(heatmap_image, (rgb_fullsz_image.shape[1], rgb_fullsz_image.shape[0]))

    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_image), cv2.COLORMAP_JET)

    overlaid_image = normalize_image((1-alpha)*rgb_fullsz_image + alpha*heatmap_colored)

    return overlaid_image



def get_test_dataloader(data_folder_path):

    # create two datsets, one with original size and one resized
    ddsm_test_fullsize = datasets.ImageFolder(root=str(data_folder_path + 'test'),
                                          transform=transforms.ToTensor())
    ddsm_test = datasets.ImageFolder(root=str(data_folder_path + 'test'),
                                          transform=transform)

    dataloader_test_fullsize = DataLoader(ddsm_test_fullsize, batch_size=1, num_workers=8)
    dataloader_test = DataLoader(ddsm_test, batch_size=1, num_workers=8)
    return dataloader_test_fullsize, dataloader_test


def save_attribution_maps(model_architecture_path, model_weights_path, data_folder_path, layer_number_list, model_name):

    # load model
    trained_model = TrainedModel(model_architecture_path, model_weights_path, model_name)

    # get dataloaders
    dataloader_fullsz, dataloader_norm = get_test_dataloader(data_folder_path)



