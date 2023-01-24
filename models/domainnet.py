import torch.nn as nn
import torch.nn.functional as F
# from .resnet import get_resnet
import torch
from torchvision import models

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


class DomainNet(nn.Module):
    def __init__(self, backbone, bn_momentum,pretrained=True, data_parallel=True):
        super(DomainNet, self).__init__()
        encoder = get_resnet(backbone,momentumn=bn_momentum,pretrained=pretrained)
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class DomainNetClassifier(nn.Module):
    def __init__(self, backbone, classes=126, data_parallel=True):
        super(DomainNetClassifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(feature_dict[backbone], classes))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        feature = torch.flatten(feature, 1)
        feature = self.linear(feature)
        return feature

class DomainnetNet(nn.Module):
    """
    Pre-trained ResNet on ImageNet with one added hidden layer, normalization,
    and activation.
    """
    def __init__(self, hidden_size=512, resnet='resnet101', pretrained=True, num_classes=345):
        super().__init__()

        if resnet == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnet == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnet == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnet == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        elif resnet == 'vgg':
            self.resnet = models.vgg16(pretrained=pretrained)

        if resnet == 'vgg':
            num_feats = self.resnet.classifier[6].in_features
            self.resnet.classifier[6] = nn.Linear(num_feats, hidden_size)
        else:
            num_feats = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_feats, hidden_size)

        
        self.linear = nn.Linear(hidden_size, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.linear(x)

        return F.log_softmax(x, dim = 1)