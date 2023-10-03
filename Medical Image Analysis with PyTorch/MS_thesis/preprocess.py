import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib.pyplot as plt

# set device to GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalization paramteters for imagenet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denormalize(tensor):
    tensor = tensor*std + mean
    return tensor

def show_img(img):
    # arrange channels
    img = img.numpy().transpose((1,2,0))

    # use mean and std values
    img = denormalize(img)

    # clip values and view image
    img = np.clip(img,0,1)
    plt.imshow(img)
    
# custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    

# returns train and test data loader objects, resizing_factor is a size tuple 
def get_data_loaders(img_path, annotations_path, resizing_factor, test_ratio=0.3, batch_size=64):

    # define transformer object
    transform = transforms.Compose([transforms.Resize(resizing_factor),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    
    # create custom dataset object
    dataset = CustomImageDataset(annotations_path, img_path, transform=transform)
    
    # train test split
    train_ratio = 1-test_ratio
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size
    
    # create data loaders
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # create dataloader objects
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
