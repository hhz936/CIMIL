# -*- coding: utf-8 -*-
import torch, time, os
import models, utils
from torch import nn, optim
from dataset import load_datasets
from config import config
from sklearn.metrics import roc_auc_score, recall_score
import numpy as np
import random
import pandas as pd
from models import resnet1d, MyNet, My02Net, My03Net
import torch
from thop import profile
# import warnings
# warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(best_acc, model, optimizer, epoch):
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints', config.model_name + '_' + config.experiment + '_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, train_dataloader, m, threshold=0.5):
    model.train()
    loss_meter, loss_meter1, loss_meter2, it_count, f1_meter, acc_meter, mAP_meter = 0, 0, 0, 0, 0, 0, 0
    outputs = []
    targets = []
    for inputs1, inputs2, target in train_dataloader:

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        # forward
        output, Loss1, Loss2 = model(inputs1, inputs2)

        loss = criterion(output, target)
        Loss = loss + m*Loss1 + m*Loss2
        Loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        loss_meter1 += Loss1
        loss_meter2 += Loss2
        it_count += 1

        output = torch.sigmoid(output)

        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    map = utils.compute_mAP(targets, outputs)
    auc = roc_auc_score(np.array(targets), np.array(outputs), multi_class='ovo')
    f1 = utils.calc_f1(targets, outputs, threshold)
    recall = utils.calc_recall(targets, outputs, threshold)

    print('train_loss: %.4f, loss: %.4f, Losss: %.4f,  macro_auc: %.4f,  f1: %.4f,  map: %.4f,  recall: %.4f' % (
        loss_meter / it_count, loss_meter1 / it_count, loss_meter2 / it_count, auc, f1, map, recall))
    return loss_meter / it_count, auc, f1, map, recall


def test_epoch(model, criterion, val_dataloader, m, threshold=0.5):
    model.eval()
    loss_meter, loss_meter1, loss_meter2, it_count, f1_meter, acc_meter, mAP_meter = 0, 0, 0, 0, 0, 0, 0
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs1, inputs2, target in val_dataloader:

            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            target = target.to(device)
            output, Loss1, Loss2 = model(inputs1, inputs2)
            loss = criterion(output, target)
            Loss = loss + m * Loss1 + m * Loss2
            loss_meter1 += Loss1
            loss_meter += loss.item()
            loss_meter2 += Loss2
            it_count += 1

            output = torch.sigmoid(output)

            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        map = utils.compute_mAP(targets, outputs)

        auc = roc_auc_score(np.array(targets), np.array(outputs),multi_class='ovo')
        f1 = utils.calc_f1(targets, outputs, threshold)
        recall = utils.calc_recall(targets, outputs, threshold)

    print('test_loss: %.4f, loss: %.4f, Losss: %.4f,  macro_auc: %.4f,  f1: %.4f,  map: %.4f,  recall: %.4f' % (
        loss_meter / it_count, loss_meter1 / it_count, loss_meter2 / it_count, auc, f1, map, recall))
    return loss_meter / it_count, auc, f1, map, recall


def train(config=config):
    # seed
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())
    # datasets
    train_dataloader, test_dataloader, num_classes = load_datasets(datafolder=config.datafolder)
    # mode
    model = getattr(My03Net, config.model_name)(num_classes=num_classes)
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    model = model.to(device)
    # optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # =========>train<=========
    for epoch in range(1, config.max_epoch + 1):
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size,
                                                                             config.lr))
        m = min(config.lamda_a2b, config.lamda_cof * epoch)
        since = time.time()

        train_loss, train_auc, train_f1, train_map, train_recall = train_epoch(model, optimizer, criterion,
                                                                               train_dataloader, m,threshold=0.5)
        # sincee = time.time()
        test_loss, test_auc, test_f1, test_map, test_recall = test_epoch(model, criterion, test_dataloader,m,
                                                                             threshold=0.5)

        save_checkpoint(test_auc, model, optimizer, epoch)

        result_list = [[epoch, train_loss, train_auc, train_f1, train_map, train_recall,
                        test_loss, test_auc, test_f1, test_map, test_recall]]

        if epoch == 1:
            columns = ['epoch', 'train_loss', 'train_auc', 'train_f1', 'train_map', 'train_recall',
                       'test_loss', 'test_auc', 'test_f1', 'test_map', 'test_recall']

        else:
            columns = ['', '', '', '', '', '', '', '', '', '', '']

        dt = pd.DataFrame(result_list, columns=columns)
        dt.to_csv(config.model_name + config.experiment + 'result.csv', mode='a')

        print('time:%s\n' % (utils.print_time_cost(since)))


if __name__ == '__main__':
    config.datafolder = 'path/'
    config.seed = 24
    train(config)
