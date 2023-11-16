import torch
import torch.nn as nn

# load a model that has been modified and trained for generating attricution maps
class TrainedModel(nn.Module):
    def __init__(self, model_arc_path, model_weights_path, name):
        super(TrainedModel, self).__init__()
        self.model = torch.load(model_arc_path)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.name = name

    def forward(self, x):
        return self.model(x)