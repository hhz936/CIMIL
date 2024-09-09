import torch
from torch import nn
import torch.nn.functional as F
import math, os, time
from CIL import CIL
from models.heartnet import heatNet
from models.resnet1d_wang import resnet1d_wang
from config import config
from torch.backends import cudnn


cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class select_class_max(nn.Module):
    def __init__(self, feature_size, output_class):
        super(select_class_max, self).__init__()
        self.fc = nn.Linear(feature_size, output_class)
        self.drop = nn.Dropout(0.)

    def forward(self, x1, x2):
        original_x1 = x1
        original_x2 = x2
        out1 = self.fc(x1)
        out1 = self.drop(out1)
        out2 = self.fc(x2)
        out2 = self.drop(out2)

        d = torch.zeros(0).cuda()
        for i in range(x1.shape[0]):
            _, m_indices = torch.sort(out1[i], 0, descending=True)
            # print(m_indices)
            a = m_indices[0, :]
            # print(a.shape)
            c = torch.zeros(0).cuda()
            for j in range(a.shape[0]):

                b = original_x1[i][a[j], :]
                # print(b.shape)
                c = torch.cat((c, b.unsqueeze(0)), 0)
            # print(c.shape)
            d = torch.cat((d, c.unsqueeze(0)), 0)
        # print(d.shape)

        d1 = torch.zeros(0).cuda()
        for i in range(x2.shape[0]):
            _, m_indices = torch.sort(out2[i], 0, descending=True)
            # print(m_indices)
            a1 = m_indices[0, :]
            # print(a.shape)
            c1 = torch.zeros(0).cuda()
            for j in range(a1.shape[0]):
                b1 = original_x2[i][a1[j], :]
                # print(b.shape)
                c1 = torch.cat((c1, b1.unsqueeze(0)), 0)
            # print(c.shape)
            d1 = torch.cat((d1, c1.unsqueeze(0)), 0)
        # print(d1.shape)

        return d, d1



class convatt(nn.Module):
    def __init__(self, dim=120):
        super(convatt, self).__init__()
        self.proj = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, 0, groups=dim), nn.GELU())
        self.proj1 = nn.Sequential(nn.Conv1d(dim, dim, 3, 1, 3 // 2, groups=dim), nn.GELU())
        self.proj2 = nn.Sequential(nn.Conv1d(dim, dim, 5, 1, 5 // 2, groups=dim), nn.GELU())
        self.proj3 = nn.Sequential(nn.Conv1d(dim, dim, 7, 1, 7 // 2, groups=dim), nn.GELU())

    def forward(self, x1):

        x = x1.transpose(1, 2)
        out = self.proj2(x)  + self.proj3(x) + self.proj1(x) + self.proj(x) + x
        out = out.transpose(1, 2)

        return out


class att_select(nn.Module):
    def __init__(self,
                 input_size, num_class, dropout_v=0.):
        super(att_select, self).__init__()
        self.q = nn.Sequential(nn.Tanh(),nn.Dropout(dropout_v), nn.Linear(input_size, input_size))
        self.selectmax = select_class_max(input_size, num_class)

    def forward(self, x1, x2):
        h1, h2 = self.selectmax(x1, x2)

        L = torch.cat((x1,x2), dim = 1)
        q_max1 = self.q(h1)  # compute queries of critical instances, q_max in shape C x Q
        A1 = torch.bmm(L, q_max1.transpose(1, 2))
        A1 = F.softmax(A1 / torch.sqrt(torch.tensor(L.shape[2], dtype=torch.float32, device=device)), 1)
        x1 = torch.bmm(A1.transpose(1, 2), L)
        x1 = 0.9*x1+0.1*q_max1
        q_max2 = self.q(h2)  # compute queries of critical instances, q_max in shape C x Q
        A2 = torch.bmm(L, q_max2.transpose(1, 2))
        A2 = F.softmax(A2 / torch.sqrt(torch.tensor(L.shape[2], dtype=torch.float32, device=device)), 1)
        x2 = torch.bmm(A2.transpose(1, 2), L)
        x2 = 0.9*x2 + 0.1*q_max2

        ATT1 = A1[:, :60, :1]
        ATT2 = A2[:, 60:, :1]
        att11 = A1[:, :60, 1:].sum(dim=2)
        att22 = A2[:, 60:, 1:].sum(dim=2)
        att12 = A1[:, 60:, 1:].sum(dim=2)
        att21 = A2[:, :60, 1:].sum(dim=2)

        x = torch.cat((x1,x2), dim = 2)

        return x, x1, x2, h1, h2, ATT1, ATT2, att11, att22, att12, att21


class OURAttention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_class,
                 proj_drop_ratio=0.):
        super(OURAttention, self).__init__()
        self.layer1 = att_select(dim, num_class)

    def forward(self, x1, x2):
        xx, XX1, XX2, h1, h2, att1, att2,att11, att22, att12, att21 = self.layer1(x1, x2)

        return xx, att1, att2, att11, att22, att12, att21



class MYMODNet(nn.Module):


    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.model1 = resnet1d_wang()
        self.model2 = heatNet()
        self.cnn = convatt(120)
        self.attention = OURAttention(120, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=num_classes*120, out_features=1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        self.dropout1 = nn.Dropout(p=0.)
        self.Apool = nn.AdaptiveAvgPool1d(120)

    def forward(self, x1, x2):
        x1 = [self.model1(x1[:, i, :, :]).view((x1.shape[0], -1, 1)) for i in range(config.n_segments1 + 1)]
        x1 = torch.cat(x1, dim=2)
        x1 = torch.transpose(x1, 1, 2)

        x2 = [self.model2(x2[:, i, :, :, :]).view((x2.shape[0], -1, 1)) for i in range(config.n_segments1 + 1)]
        x2 = torch.cat(x2, dim=2)
        x2 = torch.transpose(x2, 1, 2)

        x1 = self.cnn(x1)
        x2 = self.cnn(x2)

        out, att1, att2, att11, att22, att12, att21 = self.attention(x1, x2)
        out = self.dropout1(out)
        out = self.Apool(out)
        out = out.view(out.size(0), -1)
        y = self.decoder(out)
        loss1, loss2 = CIL(y, att1, att2, x1, x2)

        return  y, loss1, loss2, att11, att22, att12, att21


