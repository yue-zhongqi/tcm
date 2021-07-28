import torch.nn as nn
from torchvision import models


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}

class ResNet(nn.Module):
    def __init__(self, resnet_name, return_feature_map=False, frozen=[], use_max_pool=False):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.return_feature_map = return_feature_map
        self.frozen = frozen
        self.use_max_pool = use_max_pool
        if return_feature_map:
            self.layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
            self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4)
        else:
            self.layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
            if self.use_max_pool:
                self.layers.append("maxpool")
                pool = self.maxpool
            else:
                self.layers.append("avgpool")
                pool = self.avgpool
            self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, pool)
        self.n_features = model_resnet.fc.in_features

    def forward(self, x, gvbg=True):
        if len(self.frozen) == 0:
            x = self.feature_layers(x)
            if self.return_feature_map:
                return x
            else:
                x = x.view(x.size(0), -1)
                return x
        else:
            for name in self.layers:
                module = self._modules[name]
                x = module(x)
                x = x.detach() if name in self.frozen else x
            if self.return_feature_map:
                return x
            else:
                x = x.view(x.size(0), -1)
                return x
    
    def pool(self, x):
        if self.use_max_pool:
            return self.maxpool(x).view(x.size(0), -1)
        else:
            return self.avgpool(x).view(x.size(0), -1)

    def parameters(self):
        return self.feature_layers.parameters()