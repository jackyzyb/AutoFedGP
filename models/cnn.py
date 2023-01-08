import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
Adapted from PyTorch examples.
"""
class CNN(nn.Module):
    def __init__(self, hidden_size = 512):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14400, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.dropout1(x)
        # x = self.fc3(x)

        return x

class MLTCNN(nn.Module):
    def __init__(self, hidden_size = 512):
        super(MLTCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14400, hidden_size)
        self.fc_covid = nn.Linear(hidden_size, 1)
        self.fc_race = nn.Linear(hidden_size, 4)
        self.fc_sex = nn.Linear(hidden_size, 1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        pred_covid = self.fc_covid(x)
        pred_sex = self.fc_sex(x)
        pred_race = F.log_softmax(self.fc_race(x), dim=1)

        return pred_covid, pred_race, pred_sex