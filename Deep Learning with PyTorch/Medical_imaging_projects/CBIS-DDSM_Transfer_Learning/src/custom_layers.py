import torch.nn as nn

# define some contastants
VGG_OUT_CLASSIFIER = 4096     # output dim of the final fc layer
VGG_OUT_FEATURES = 7*7*512    # output dim of the last conv block
VGG_OUT_CHANNELS = 512

RESNET34_OUT_FEATURES = 1*1*512
RESNET50_OUT_FEATURES = 1*1*2048
EFFICIENTNETB6_OUT_FEATURES = 1*1*2304
EFFICIENTNETB7_OUT_FEATURES = 1*1*2560

def get_layers(model_name, num_classes, modification_type):

    custom_layers = dict()
    if model_name in ['vgg16', 'vgg19']:
        if modification_type == 'final_fc':

            custom_layers['final_classifier'] =  nn.Sequential(nn.Linear(VGG_OUT_CLASSIFIER, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, num_classes),
                                nn.LogSoftmax(dim=1))


        elif modification_type == 'complete_fc':

            custom_layers['final_features'] = nn.Sequential(nn.Conv2d(VGG_OUT_CHANNELS, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                         nn.ReLU(),
                                         nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

            custom_layers['complete_classifier'] = nn.Sequential(nn.Linear(7 * 7 * 128, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(256, num_classes),
                                         nn.LogSoftmax(dim=1))


    elif model_name in ['resnet34', 'resnet50']:
        if modification_type == 'final_fc' and model_name =='resnet34':
            custom_layers['complete_fc'] = nn.Sequential(nn.Linear(RESNET34_OUT_FEATURES, num_classes),
                                           nn.LogSoftmax(dim=1))

        if modification_type == 'final_fc' and model_name == 'resnet50':
            custom_layers['complete_fc'] = nn.Sequential(nn.Linear(RESNET50_OUT_FEATURES, num_classes),
                                                             nn.LogSoftmax(dim=1))

    elif model_name in ['efficientnet_b6', 'efficientnet_b7']:
        if modification_type == 'final_fc' and model_name == 'efficientnet_b6':
            custom_layers['complete_fc'] = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                           nn.Linear(EFFICIENTNETB6_OUT_FEATURES, num_classes),
                                           nn.LogSoftmax(dim=1))

        if modification_type == 'final_fc' and model_name == 'efficientnet_b7':
            custom_layers['complete_fc'] = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                                        nn.Linear(EFFICIENTNETB7_OUT_FEATURES, num_classes),
                                                        nn.LogSoftmax(dim=1))

    return custom_layers