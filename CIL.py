import torch, os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from InfoNCE import InfoNCE
from InfoNCE import InfoNCE


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def CIL(logits, att1, att2, time_rep, visual_rep):
    logitss = torch.sigmoid(logits)

    time_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()
    time_abn = torch.zeros(0).cuda()
    visual_abn = torch.zeros(0).cuda()

    for i in range(logitss.shape[0]):
        if logitss[i][0] > 0.5:
            time_inverse_topk, time_topk_indices = torch.topk(att1[i], k=3, dim=0,  largest=True)
            # print(time_topk_indices.shape)
            time_nor_topk = time_rep[i][time_topk_indices]
            # print(time_nor_topk.shape)
            time_nor = torch.cat((time_nor, time_nor_topk), 0)

            # print(time_nor.shape)

            visual_inverse_topk, visual_topk_indices = torch.topk(att2[i], k=3, dim=0,largest=True)
            # print(time_topk_indices.shape)
            visual_nor_topk = visual_rep[i][visual_topk_indices]
            # print(time_nor_topk.shape)
            visual_nor = torch.cat((visual_nor, visual_nor_topk), 0)

            # print(visual_nor.shape)
        else:
            time_inverse_topk, time_topk_indices = torch.topk(att1[i], k=3, dim=0,largest=False)
            # print(time_topk_indices.shape)
            time_abn_topk = time_rep[i][time_topk_indices]
            # print(time_nor_topk.shape)
            time_abn = torch.cat((time_abn, time_abn_topk), 0)
            # print(time_abn.shape)

            visual_inverse_topk, visual_topk_indices = torch.topk(att2[i], k=3, dim=0,largest=False)
            # print(time_topk_indices.shape)
            visual_abn_topk = visual_rep[i][visual_topk_indices]
            # print(time_nor_topk.shape)
            visual_abn = torch.cat((visual_abn, visual_abn_topk), 0)
            # print(visual_abn.shape)

    cils = InfoNCE()
    if time_nor.size(0) == 0 or time_abn.size(0) == 0 or visual_nor.size(0) == 0 or visual_abn.size(0) == 0:
        return 0.0, 0.0
    else:
        loss1 = cils(time_abn, visual_abn, time_nor)
        loss2 = cils(visual_abn, time_abn, visual_nor)
        # loss3 = cils(time_abn, visual_abn, time_nor)
        return loss1, loss2


if __name__ == '__main__':

    audio_logits = torch.randn(64, 6)
    # att = torch.randn(64, 1, 60)
    time_rep = torch.randn(64, 60, 120)
    visual_rep = torch.randn(64, 60, 120)

    att1 = torch.randn(64, 60, 1)
    att2 = torch.randn(64, 60, 1)

    audio_logits = audio_logits.to(device)
    # att = att.to(device)
    time_rep = time_rep.to(device)
    visual_rep = visual_rep.to(device)
    a, b= CIL(audio_logits, att1, att2, time_rep, visual_rep)
    print(a)
    print(b)
    # print(c)

