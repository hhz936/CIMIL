import torch
import torch.nn.functional as F
from torch import nn, transpose
import numpy as np

__all__ = ['InfoNCE', 'info_nce']


def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x.cpu().detach().numpy()) * np.linalg.norm(y.cpu().detach().numpy())
    return num / denom


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.92):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,)


def info_nce(query, positive_key, negative_keys, temperature):
    query = query.squeeze(1)
    positive_key = positive_key.squeeze(1)
    negative_keys = negative_keys.squeeze(1)

    loss, loss_meter = 0, 0
    for i in range(query.shape[0]):
        query1 = query[i].unsqueeze(0)
        sim, sim_meter = 0, 0
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim1 = cos_sim(query1, positive_key[i].unsqueeze(0))
        sim_1 = torch.exp(sim1 / temperature)
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim2 = cos_sim(query1.repeat(negative_keys.shape[0], 1), negative_keys)
        simmm2 = torch.exp(sim2 / temperature)
        sim_2 = torch.sum(simmm2, 0)
        sim_2 = sim_2 / negative_keys.shape[0]
        loss_1 = -torch.log(sim_1 / (sim_1 + sim_2))
        loss += loss_1
        loss_meter += 1
    Loss = loss / loss_meter
    return Loss

