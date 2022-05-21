"""
Adapted from
https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STNkd(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        bs = input.size(0)
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        pool = nn.MaxPool1d(x.size(-1))(x)
        flat = nn.Flatten(1)(pool)
        x = F.relu(self.bn4(self.fc1(flat)))
        x = F.relu(self.bn5(self.fc2(x)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        init = init.to(device)
        matrix = self.fc3(x).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = STNkd(k=3)
        self.feature_transform = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        x = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))

        matrix64x64 = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(x)
        return output, matrix3x3, matrix64x64


class PointNetFc(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, momentum=None):
        x, matrix3x3, matrix64x64 = self.transform(input)

        # Attach moemntum
        if momentum:
            x = torch.hstack([x, momentum.unsqueeze(1)])
            self.fc1 = nn.Linear(1024 + 1, 512)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.fc3(x)
        return self.logsoftmax(output), matrix3x3, matrix64x64
