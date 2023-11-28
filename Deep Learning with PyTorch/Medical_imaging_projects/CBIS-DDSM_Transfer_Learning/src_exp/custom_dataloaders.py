import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

def get_dataloaders(data_folder, batch_size, val_ratio):
    mean = [0.42956483, 0.42956483, 0.42956483]
    std = [0.09959752, 0.09959752, 0.09959752]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ddsm_train = datasets.ImageFolder(root=str(data_folder + 'train'),
                                      transform=transform)

    labels = [label for _, label in ddsm_train]
    train_idxs, val_idxs = train_test_split(np.arange(len(ddsm_train)),
                                            test_size=val_ratio,
                                            random_state=42,
                                            shuffle=True,
                                            stratify=labels)

    # load test dataset
    ddsm_test = datasets.ImageFolder(root=str(data_folder + 'test'), transform=transform)
    print(f'Class_indices testdata {ddsm_test.class_to_idx}')

    dl_train = DataLoader(Subset(ddsm_train, train_idxs), batch_size, shuffle=True, num_workers=2)
    dl_val = DataLoader(Subset(ddsm_train, val_idxs), batch_size, shuffle=False, num_workers=2)
    dl_test = DataLoader(ddsm_test, batch_size, num_workers=2)

    return dl_train, dl_val, dl_test