import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.iden = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))
        ).view(1, -1)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))

        # initialize as identity
        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize, 1, 1)
        iden = iden.to(device)
        x = self.fc3(x).view(-1, self.k, self.k) + iden

        return x


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = STNkd(k=3)
        self.feature_transform = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):

        # batch matrix multiplication
        # input transform
        x = torch.bmm(
            torch.transpose(x, 1, 2), self.input_transform(x)
        ).transpose(1, 2)

        x = self.relu(self.bn1(self.conv1(x)))

        # Feature transform
        x = torch.bmm(
            torch.transpose(x, 1, 2), self.feature_transform(x)
        ).transpose(1, 2)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.view(-1, 1024)
        return x


class PointNetFc(nn.Module):
    def __init__(
        self,
        num_classes,
        momentum=False,
        radius=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.radius = radius

        self.feat = Transform()

        # include radius and momentum
        if self.momentum and self.radius:
            self.fc1 = nn.Linear(1024 + 2, 512)
        elif self.momentum or self.radius:
            self.fc1 = nn.Linear(1024 + 1, 512)
        else:
            self.fc1 = nn.Linear(1024, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.dropout = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x, momentum=None, radius=None):

        x = x.transpose(1, 2)

        x = self.feat(x)

        # add momentum dimension if desired
        if self.momentum and momentum is not None:
            x = torch.hstack([x, momentum.unsqueeze(1)])

        # add radius dimension if desired
        if self.radius and radius is not None:
            x = torch.hstack([x, radius.unsqueeze(1)])

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x
