<<<<<<< HEAD
=======
"""Adapted from watchML 
"""

>>>>>>> pointnet v2
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
<<<<<<< HEAD

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

=======


class PointMaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.max(x, 2, keepdim=True)[0]
        return x


class PointMeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, 2, keepdim=True)
        return x


class PointMeanMaxPool(nn.Module):
    def __init__(self, mean_pool=0):
        super().__init__()
        self.mean_pool_split = mean_pool

    def forward(self, x):
        x1 = torch.mean(x[:, : self.mean_pool_split, :], 2, keepdim=True)
        x2 = torch.max(x[:, self.mean_pool_split :, :], 2, keepdim=True)[0]
        x = torch.cat((x1, x2), 1)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, mean_pool=0, max_feat=256):
        super().__init__()
        self.max_feat = max_feat
        min_feat = 256  # 1 << (k * k - 1).bit_length()
        n_feat2 = max(max_feat // 2, min_feat)
        n_feat3 = max(n_feat2 // 2, min_feat)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, max_feat, 1)
        self.fc1 = nn.Linear(max_feat, n_feat2)
        self.fc2 = nn.Linear(n_feat2, n_feat3)
        self.fc3 = nn.Linear(n_feat3, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(max_feat)
        self.bn4 = nn.BatchNorm1d(n_feat2)
        self.bn5 = nn.BatchNorm1d(n_feat3)

        self.k = k
        if mean_pool <= 0:
            self.pool = PointMaxPool()
        elif mean_pool >= max_feat:
            self.pool = PointMeanPool()
        else:
            self.pool = PointMeanMaxPool(mean_pool)

        self.iden = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))
        ).view(1, -1)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, self.max_feat)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    def __init__(
        self, feature_transform=False, k=5, mean_pool=0, num_output_channels=256
    ):
        super().__init__()
        self.max_feat = num_output_channels
        self.stn = STNkd(k=k, mean_pool=mean_pool, max_feat=num_output_channels)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, num_output_channels, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(num_output_channels)
        self.feature_transform = feature_transform
        if mean_pool <= 0:
            self.pool = PointMaxPool()
        elif mean_pool >= num_output_channels:
            self.pool = PointMeanPool()
        else:
            self.pool = PointMeanMaxPool(mean_pool)
        if self.feature_transform:
            self.fstn = STNkd(k=64, mean_pool=mean_pool, max_feat=num_output_channels)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.max_feat)
        return x


class PointNetFeedForward(nn.Module):
    
    def __init__(self, k):
        super().__init__()

        self.feat = PointNetFeat(k=3, num_output_channels=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
>>>>>>> pointnet v2
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

<<<<<<< HEAD
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
=======
    def forward(self, x, p):

        x = self.feat(x)

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        return x
>>>>>>> pointnet v2
