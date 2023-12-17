import torch.nn as nn

# define some contastants
VGG_OUT_CHANNELS = 512
RESNET50_OUT_FEATURES = 2048


def get_layers(model_name, num_classes, modification_type):
    custom_layers = dict()
    if model_name == 'vgg19' and modification_type == 'final_fc':
        custom_layers['final_classifier'] = nn.Sequential(nn.Linear(VGG_OUT_CHANNELS, num_classes),
                                                          nn.LogSoftmax(dim=1))


    elif model_name == 'resnet50' and modification_type == 'final_fc':
        custom_layers['complete_fc'] = nn.Sequential(nn.Linear(RESNET50_OUT_FEATURES, num_classes),
                                                     nn.LogSoftmax(dim=1))

    return custom_layers