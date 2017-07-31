import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.mp1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(8)
        self.mp2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.mp3 = nn.AvgPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 8 * 8, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.mp3(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)

        x = x.view(-1, 16 * 8 * 8)
        x = self.fc(x)
        return x
