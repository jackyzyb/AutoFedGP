import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from domainbed import networks

class domainbedNet(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(domainbedNet, self).__init__()
        self.hparams = hparams
        featurizer = networks.Featurizer(input_shape, self.hparams)
        classifier = networks.Classifier(
            featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(featurizer, classifier)

    def forward(self, x):
        x = self.network(x)
        return x
