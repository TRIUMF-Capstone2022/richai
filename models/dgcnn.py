"""
Adapted from:
https://github.com/WangYueFt/dgcnn (Author implementation)
https://github.com/AnTao97/dgcnn.pytorch (Author suggested alternative implementation)
https://dl.acm.org/doi/pdf/10.1145/3326362 (Original paper)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def knn(x, k):
    """KNN implementation for finding graph neighbors."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k, idx=None):
    """Dynamically calculate graph edge features."""
    batch_size, num_dims, num_points = x.size()

    # find knn
    if idx is None:
        idx = knn(x, k=k)

    # index for each point: (batch_size) -> view -> (batch_size, 1, 1)
    idx_base = (
        torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)
        * num_points
    )

    # index + knn index: (batch_size, num_dims, num_points)
    idx = idx + idx_base

    # (batch_size * num_dims * num_points)
    idx = idx.view(-1)

    # (batch_size, num_dims, num_points) -> (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()

    # (batch_size * num_points * k, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]

    # (batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)

    # (batch_size, num_points, 1, num_dims) -> repeat -> (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # (batch_size, 2*num_dims, num_points, k)
    feature = (
        torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    )

    return feature


class DGCNN(nn.Module):
    """Dynamic Graph CNN

    Attributes
    ----------
    input_channels : int
       Default input features are 3 coordinates * 2 because of edge features
    output_channels : int
       Output channels
    k : int
        k neighbors
    dropout : float
        Regularization dropout parameters
    momentum : bool
        If true, include momentum as feature
    radius : bool
        If true, include radius as feature

    Methods
    -------
    forward(x, p, radius)
        Feed forward layer with input x, momentum and radius
    """

    def __init__(
        self,
        input_channels=6,
        output_channels=3,
        k=16,
        dropout=0.5,
        momentum=False,
        radius=False,
    ):
        super(DGCNN, self).__init__()

        # default input features are 3 coordinates * 2 because of edge features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.k = k
        self.dropout = dropout
        self.momentum = momentum
        self.radius = radius

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        if self.momentum and self.radius:
            self.linear1 = nn.Linear(1024 * 2 + 2, 512, bias=False)
        elif self.momentum or self.radius:
            self.linear1 = nn.Linear(1024 * 2 + 1, 512, bias=False)
        else:
            self.linear1 = nn.Linear(1024 * 2, 512, bias=False)

        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        self.linear3 = nn.Linear(256, self.output_channels)

    def forward(self, x, p=None, radius=None):
        # X needs to be reshaped for knn calculation to work
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x, k=self.k)

        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)

        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x1, k=self.k)

        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)

        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2, k=self.k)

        # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x = self.conv3(x)

        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = get_graph_feature(x3, k=self.k)

        # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x = self.conv4(x)

        # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64+64+128+256, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # (batch_size, 64+64+128+256, num_points) -> (batch_size, 1024, num_points)
        x = self.conv5(x)

        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)

        # (batch_size, 1024*2 + mom + rad)
        x = torch.cat((x1, x2), 1)

        # add momentum dimension if desired
        if self.momentum and p is not None:
            x = torch.hstack([x, p.unsqueeze(1)])

        # add radius dimension if desired
        if self.radius and radius is not None:
            x = torch.hstack([x, radius.unsqueeze(1)])

        # (batch_size, 1024*2 + mom + rad) -> (batch_size, 512)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)

        # (batch_size, 512) -> (batch_size, 256)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)

        # (batch_size, 256) -> (batch_size, output_channels)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
