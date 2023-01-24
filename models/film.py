import copy

import torch
from torch import nn
from torchvision import models


class FiLM(nn.Module):
    """
    Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`
    (https://arxiv.org/abs/1709.07871).
    """
    def __init__(self, output_size=2048):
        super().__init__()

        self.register_parameter(name='gamma',
                                param=nn.Parameter(torch.ones(output_size)))
        self.register_parameter(name='beta',
                                param=nn.Parameter(torch.zeros(output_size)))

    def forward(self, x):
        """
        Forward pass (modulation)

        :param input: input features (N, C, *) where * is any number of dims
        :param gamma: (C,)
        :param beta: (C,)
        :return: output, modulated features (N, C, *), same shape as input
        """
        gamma = self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = self.beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gamma * x) + beta


class FiLMedResNetModular(nn.Module):
    """
    ResNet-152 with FiLM layers introduced before each major block in the
    ResNet architecture (before the convolutional blocks). Its initialization
    takes in a list of numbers corresponding to locations where the FiLM layers
    are introduced.
    """
    def __init__(self, hidden_size=1024, film_layers=[]):
        super().__init__()

        self.resnet = models.resnet152(pretrained=True)

        layer2_input = self.resnet.layer2[0].conv1.in_channels
        layer3_input = self.resnet.layer3[0].conv1.in_channels
        layer4_input = self.resnet.layer4[0].conv1.in_channels

        self.film1 = FiLM(output_size=layer2_input)
        self.film2 = FiLM(output_size=layer3_input)
        self.film3 = FiLM(output_size=layer4_input)

        self.resnet_body = nn.ModuleList()
        self.resnet_body.append(self.resnet.layer1)
        if 1 in film_layers:
            self.resnet_body.append(self.film1)
        self.resnet_body.append(self.resnet.layer2)
        if 2 in film_layers:
            self.resnet_body.append(self.film2)
        self.resnet_body.append(self.resnet.layer3)
        if 3 in film_layers:
            self.resnet_body.append(self.film3)

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for layer in self.resnet_body:
            x = layer(x)

        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.squeeze()

        x = self.leaky_relu(self.resnet.fc(x))
        x = self.leaky_relu(self.linear(x))

        return x


class FiLMedResNetBN(nn.Module):
    """
    ResNet-152 with FiLM layers introduced as replacements to some of the
    ResNet's batch normalization layers.
    """
    def __init__(self, hidden_size=1024, replace_mode='sparse',
                 replace_layers=[], bn_replace=[3]):
        super().__init__()

        if replace_mode == 'custom':
            assert replace_layers and bn_replace
        else:
            assert not replace_layers

        self.resnet = models.resnet152(pretrained=True)

        layer2_input = self.resnet.layer2[0].conv1.in_channels
        layer3_input = self.resnet.layer3[0].conv1.in_channels
        layer4_input = self.resnet.layer4[0].conv1.in_channels

        if replace_mode == 'sparse':

            # Replace layer1 final bn3
            self.resnet.layer1._modules['2']._modules['bn3'] = \
                FiLM(output_size=layer2_input)

            # Replace layer2 final bn3
            self.resnet.layer2._modules['7']._modules['bn3'] = \
                FiLM(output_size=layer3_input)

            # Replace layer3 final bn3
            self.resnet.layer3._modules['35']._modules['bn3'] = \
                FiLM(output_size=layer4_input)

        elif replace_mode == 'all':
            for name, layer in self.named_modules():
                # Replace all batch norm layers before a ReLU with FiLM
                if isinstance(layer, torch.nn.BatchNorm2d):
                    try:  # Check if batch norm is in one of the main layers
                        _, layer_num, block_num, bn_num = name.split('.')
                        if int(bn_num[-1]) in bn_replace:
                            output_size = self.resnet._modules[
                                layer_num]._modules[
                                block_num]._modules[bn_num].weight.size(0)
                            self.resnet._modules[layer_num]._modules[
                                block_num]._modules[bn_num] = \
                                FiLM(output_size=output_size)
                    except ValueError:  # Catch exceptions from parsing name
                        continue

        elif replace_mode == 'custom':
            for name, layer in self.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    try:
                        _, layer_num, block_num, bn_num = name.split('.')
                        if int(layer_num[-1]) in \
                                [int(x) for x in replace_layers] and \
                                int(bn_num[-1]) in bn_replace:
                            if 3.1 in replace_layers and \
                                    int(block_num) >= 0 and \
                                    int(block_num) < 12:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                            if 3.2 in replace_layers and \
                                    int(block_num) >= 12 and \
                                    int(block_num) < 24:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                            if 3.3 in replace_layers and \
                                    int(block_num) >= 24 and \
                                    int(block_num) < 36:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                            if float(layer_num[-1]) in replace_layers:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                    except ValueError:
                        continue

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.leaky_relu(self.linear(x))

        return x


class FiLMedResNetFineTune(nn.Module):
    """
    ResNet-152 with FiLM layers introduced only during the fine-tuning phase.
    The FiLM layers are added according to the FiLMedResNet class passed during
    initialization.

    Note: Currently only supports FiLMedResNetBN
    """
    def __init__(self, pretrained: nn.Module, new_model_class: nn.Module,
                 **kwargs):
        super().__init__()

        # Initialize new model class
        new_model = new_model_class(**kwargs)
        own_state = new_model.state_dict()

        # Avoid parameter override of FiLM layers with BN weights/bias
        other_state = pretrained.state_dict()
        if isinstance(new_model, FiLMedResNetBN):
            for name, param in other_state.items():
                if isinstance(param, nn.Parameter):
                    param = param.data
                # check if BN parameters are found in model
                try:
                    own_state[name].copy_(param)
                except KeyError:
                    assert 'bn' in name
                    continue

        self.resnet = copy.deepcopy(new_model.resnet)
        self.linear = copy.deepcopy(new_model.linear)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        del new_model

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.leaky_relu(self.linear(x))

        return x


class FiLMedResNetInitialLayers(nn.Module):
    """
    ResNet-152 with FiLM layers introduced near the input layer, replacing the
    first batch norm layer (bn1).
    """
    def __init__(self, hidden_size=1024):
        super().__init__()

        self.resnet = models.resnet152(pretrained=True)

        out_channels = self.resnet.conv1.out_channels
        self.film = FiLM(output_size=out_channels)

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.film(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.squeeze()

        x = self.leaky_relu(self.resnet.fc(x))
        x = self.leaky_relu(self.linear(x))

        return x
