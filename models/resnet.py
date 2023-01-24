from torch import nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """
    Pre-trained ResNet on ImageNet with one added hidden layer, normalization,
    and activation.
    """
    def __init__(self, hidden_size=128, resnet='resnet18', pretrained=True, num_classes=10):
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
        x = self.leaky_relu(self.linear(x))

        return x


def ResNetOrig(resnet='resnet152'):
    """
    Initializs a pre-trained ResNet on ImageNet (same as torchvision.models).
    """
    if resnet == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif resnet == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif resnet == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif resnet == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif resnet == 'resnet152':
        model = models.resnet152(pretrained=True)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, 1)
    return model

if __name__ == '__main__':
    model = ResNetClassifier(hidden_size=128, resnet='resnet18', pretrained=True)
    print(sum(p.numel() for p in model.parameters()))
