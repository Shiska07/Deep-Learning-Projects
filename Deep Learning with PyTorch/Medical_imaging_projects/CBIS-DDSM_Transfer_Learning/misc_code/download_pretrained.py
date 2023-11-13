import os
import torch
from torchvision import models
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_models_path = 'pretrained_models'
models_to_download = ['vgg16', 'vgg19']


def download_pretrained_models(model_name, dst_path, use_gpu=True):
    os.makedirs(dst_path, exist_ok=True)

    if use_gpu and torch.cuda.is_available():
        model = getattr(models, model_name)(pretrained=True).cuda()
    else:
        model = getattr(models, model_name)(pretrained=True)

    save_filename = os.path.join(dst_path, f'{model_name}.pth')
    torch.save(model.state_dict(), save_filename)

    print(f'Model {model_name} download complete!')


if __name__ == "__main__":
    for model_name in models_to_download:
        download_pretrained_models(model_name, pretrained_models_path, True)