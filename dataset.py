import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data_process import preprocess_signals, data_slice, transtoimage, slicepatch, STPEtranstoimage, STpeter, \
    interSTpeter, MITspra, interMitspra
from config import config


class ECGDataset(Dataset):
    def __init__(self, signals:np.ndarray, images:np.ndarray, labels: np.ndarray):
        super(ECGDataset, self).__init__()
        self.data1 = signals
        self.data2 = images
        self.label = labels
        self.num_classes = self.label.shape[1]

        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = self.label[index]

        #x1 = x1.transpose()
        x1 = torch.tensor(x1.copy(), dtype=torch.float)


        x2 = torch.tensor(x2.copy(), dtype=torch.float)
        # x2 = x2.transpose(0, 2)
        # x2 = x2.float()

        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return  x1, x2, y

    def __len__(self):
        return len(self.data2)



def load_datasets(datafolder=None):

    if datafolder == 'C:/Users/num4/Desktop/datapre/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/':
        X_train1, X_train2, y_train, X_test1, X_test2, y_test = interSTpeter()
        X_train2 = STPEtranstoimage(X_train2)
        X_test2 = STPEtranstoimage(X_test2)
        X_train2 = slicepatch(X_train2)
        X_test2 = slicepatch(X_test2)

    elif datafolder == 'C:/Users/num4/Desktop/datapre/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/':
        X_train1, X_train2, y_train, X_test1, X_test2, y_test = STpeter()
        X_train2 = STPEtranstoimage(X_train2)
        X_test2 = STPEtranstoimage(X_test2)
        X_train2 = slicepatch(X_train2)
        X_test2 = slicepatch(X_test2)

    elif datafolder == 'C:/Users/num4/Desktop/datapre/mit-bih-supraventricular-arrhythmia-database-1.0/':
        X_train1, X_train2, y_train, X_test1, X_test2, y_test = MITspra()
        X_train2 = STPEtranstoimage(X_train2)
        X_test2 = STPEtranstoimage(X_test2)
        X_train2 = slicepatch(X_train2)
        X_test2 = slicepatch(X_test2)

    elif datafolder == 'C:/Users/num4/Desktop/datapre/mit-bih-supraventricular-arrhythmia-database-1.0/inter/':
        X_train1, X_train2, y_train, X_test1, X_test2, y_test = interMitspra()
        X_train2 = STPEtranstoimage(X_train2)
        X_test2 = STPEtranstoimage(X_test2)
        X_train2 = slicepatch(X_train2)
        X_test2 = slicepatch(X_test2)

    ds_train = ECGDataset(X_train1, X_train2, y_train)
    ds_test = ECGDataset(X_test1, X_test2, y_test)

    num_classes = ds_train.num_classes
    train_dataloader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, test_dataloader, num_classes
