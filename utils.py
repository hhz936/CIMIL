# -*- coding: utf-8 -*-
'''
@ author:
'''
import time
import math
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, f1_score, accuracy_score, \
    fbeta_score, recall_score
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.keras.utils.np_utils import to_categorical


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class model_TeacherHead(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.attention_1 = nn.Sequential(
            nn.Linear(6, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=6, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_class)
        )

    def forward(self, fea1):
        attention_1 = self.attention_1(fea1)
        attention_1 = F.softmax(attention_1, dim=1)
        A_1 = torch.transpose(attention_1, 1, 2)
        M_1 = torch.bmm(A_1, fea1)
        M_1 = torch.squeeze(M_1, dim=1)
        output = self.decoder(M_1)
        # output = torch.sigmoid(output)
        return output


class Pseudo_instance_labelHead(nn.Module):
    def __init__(self, num_class):
        super(Pseudo_instance_labelHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_class)
        )
    def forward(self, fea1):
        output = self.decoder(fea1)
        # output = torch.softmax(output, 2)
        # output = output.argmax(dim=2).unsqueeze(dim=2)
        # output = output.reshape(-1, 1)
        # output = output.squeeze()
        return output


class model_StudentHead(nn.Module):
    def __init__(self, num_class):
        super(model_StudentHead, self).__init__()
        self.num_class = num_class
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_class)
        )

    def forward(self, fea1):
        output = self.decoder(fea1)
        # output = torch.softmax(output, 2)
        return output


def Criterion_ins(output, target0):
    criterion = nn.CrossEntropyLoss()
    it_count = 0
    # Loss = torch.zeros(1)
    # for i in range(output.shape[1]):
    #     loss = criterion(output[:, i, :], target0[:, i, :].squeeze().long())
    #     Loss = Loss + loss.item()
    #     it_count += 1
    # Loss = Loss / it_count

    # for i in output:
    #     for j in target0:
    #         j = j.squeeze()
    #         loss = criterion(i, j.long())
    #         Loss = Loss + loss.item()
    #         it_count += 1
    # Loss = Loss / it_count

    output1 = output.reshape(-1, 6)
    # print(output1.shape)
    # target01 = target0.reshape(-1, 1)
    # target01 = target01.squeeze()
    # print(target01.shape)
    Loss = criterion(output1, target0.long())
    return Loss


def compute_mAP(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i], average="samples"))
    return np.mean(AP)


def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1

    return sum / count


def compute_AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auc = []
    for i in range(len(y_true[1])):
        class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = roc_auc_score(y_true, y_pred)
    return auc, class_auc



def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    y_pred_binary = apply_thresholds(y_pre, threshold)
    return f1_score(y_true, y_pred_binary, average="samples")


def calc_recall(y_true, y_pre, threshold=0.5):
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    y_pred_binary = apply_thresholds(y_pre, threshold)
    return recall_score(y_true, y_pred_binary, average="samples")


# PRINT TIME
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)


def cal_accuracy_score(y_true, y_pre, threshold=0.5):

    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_pre, y_true)


# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss


def apply_thresholds(preds, threshold):
    tmp = []
    for p in preds:
        tmp_p = (p > threshold).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp


def metrics(y_true, y_pred, beta1=2):
    f_beta = 0.
    acc = 0.
    sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:, classi], y_pred[:, classi]
        TP, FP, TN, FN = 0., 0., 0., 0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i] == y_predi[i] == 1:
                TP += 1. / sample_weight
            if (y_predi[i] == 1) and (y_truei[i] != y_predi[i]):
                FP += 1. / sample_weight
            if y_truei[i] == y_predi[i] == 0:
                TN += 1. / sample_weight
            if (y_predi[i] == 0) and (y_truei[i] != y_predi[i]):
                FN += 1. / sample_weight

        acc_i = 0.
        try:
            acc_i = (TP + TN) / (TP + TN + FP + FN)
        except ZeroDivisionError:
            pass
        acc += acc_i

        f_beta_i = 0.
        try:
            f_beta_i = ((1 + beta1 ** 2) * TP) / ((1 + beta1 ** 2) * TP + FP + (beta1 ** 2) * FN)
        except ZeroDivisionError:
            pass
        f_beta += f_beta_i

    return f_beta / y_true.shape[1], acc / y_true.shape[1]


def evaluate_metrics(y_true, y_pred, threshold=None):
    macro_auc = 0.

    if len(list(y_true.shape)) == 1:
        try:
            y_true1 = to_categorical(y_true, num_classes=None)
            macro_auc = roc_auc_score(y_true1, y_pred)
        except ValueError:
            pass
        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1_scores = fbeta_score(y_true, y_pred, beta=2, average='macro')
    else:
        try:
            macro_auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            pass

        y_pred_binary = apply_thresholds(y_pred, threshold)
        f1_scores, acc = metrics(y_true, y_pred_binary, beta1=2)

    return macro_auc, f1_scores, acc



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


