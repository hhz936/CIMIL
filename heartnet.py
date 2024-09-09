import torch.nn as nn
import torch.nn.functional as F
import torch


class HeartNet(nn.Module):
    def __init__(self, num_classes=7):
        super(HeartNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(12, eps=0.001),
            nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(32, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            # nn.MaxPool2d(kernel_size=8, stride=1),
        )
        # self.lin = nn.Linear(256*33, 120)
        self.dropout = nn.Dropout(0.)
        self.Adp = nn.AdaptiveAvgPool2d(1)

        self.lin = nn.Linear(256, 120)

    def forward(self, x):
        x = self.features(x)
        # x = self.dropout(x)
        x = self.Adp(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        # x = self.classifier(x)
        return x

def heatNet():
    return HeartNet(120)

