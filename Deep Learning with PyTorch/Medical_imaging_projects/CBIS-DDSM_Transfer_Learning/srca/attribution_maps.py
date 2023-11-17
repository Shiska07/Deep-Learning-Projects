import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils import CustomTestDataset, mean, std, normalize_image
from captum.attr import LayerGradCam, LayerAttribution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])

class TrainedModel(nn.Module):
    def __init__(self, model_arc_path, model_weights_path, name):
        super(TrainedModel, self).__init__()
        self.model = torch.load(model_arc_path)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.name = name

    def forward(self, x):
        return self.model(x)


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


def get_test_dataloader(image_folder_path, annotations_file_path):

    # create two datsets, one with original size and one resized
    ddsm_test_fullsize = CustomTestDataset(image_folder_path, annotations_file_path,
                                          transform=transforms.ToTensor())
    ddsm_test = CustomTestDataset(image_folder_path, annotations_file_path,
                                          transform=transform)

    dataloader_test_fullsize = DataLoader(ddsm_test_fullsize, batch_size=1, num_workers=8)
    dataloader_test = DataLoader(ddsm_test, batch_size=1, num_workers=8)
    return dataloader_test_fullsize, dataloader_test


def save_attribution_maps(model_architecture_path, model_weights_path, data_folder_path, layer_number, model_name):

    # load model
    trained_model = TrainedModel(model_architecture_path, model_weights_path, model_name)

    # get dataloaders
    test_annotations_path = os.path.join(data_folder_path, 'annotations.csv')
    test_images_path = os.path.join(data_folder_path, 'allimages')
    dataloader_fullsz, dataloader_norm = get_test_dataloader(test_images_path, test_annotations_path)

    # create destination folder
    dest_folder_path = os.path.join(data_folder_path, 'atribution_maps')
    if not os.path.exists(data_folder_path):
        os.makedirs(dest_folder_path)

    # iterate through images and save attrbution maps
    for vals1, vals2 in zip(dataloader_fullsz, dataloader_norm):
        imagef, labelf, _ = vals1
        imagen, labeln, img_name = vals2

        attribution_map = get_attention_map(trained_model, imagen[0], imagef[0], layer_number, 0.3)
        attribution_map_path = os.path.join(dest_folder_path, img_name)
        cv2.imwrite(attribution_map_path, attribution_map)

    print(f'Attribution maps for {model_name} layer {layer_number} saved at {dest_folder_path}.\n')

