import os
import torch
import numpy as np
from einops import rearrange
from tqdm import tqdm
import wfdb
import ast
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from pyts.image import GramianAngularField


def name2index(path):
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def file2index(path, name2idx):
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        file2index[id] = labels
    return file2index


def preprocess_signals(X_train, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    return apply_standardizer(X_train, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def data_slice(data):
    data_process = []
    for dat in data:
        if dat.shape[0] < 1000:
            # dat = np.pad(dat, (0, 1000 - dat.shape[0]), 'constant', constant_values=0)
            dat = resample(dat, 1000, axis=0)
        elif dat.shape[0] > 1000:
            dat = dat[:1000, :]
            # dat = resample(dat, 1000, axis=0)
        if dat.shape[1] != 12:
            dat = dat[:, 0:12]

        data_process.append(dat)
    return np.array(data_process)


def dataresamp(data):
    data_process = []
    for dat in data:

        dat = resample(dat, int(dat.shape[0]*0.6), axis=0)

        if dat.shape[1] != 12:
            dat = dat[:, 0:12]

        data_process.append(dat)
    return np.array(data_process)


def slicepatch(data):
    data = rearrange(data, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=12, p2=45)
    data = data.numpy()
    return data

#STpeter  instance data
def preprosessSTpeter():
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/headatatr/']:
        filenames = os.listdir('C:/Users/num4/Desktop/datapre/' + folder)
        data = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('C:/Users/num4/Desktop/datapre/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('C:/Users/num4/Desktop/datapre/' + folder + name, "atr", sampfrom=0,
                                               sampto=462600)
                for i in range(60 * (len(signal_annotation.sample) // 60 - 1)+1):
                    ventricular_signal = ECGdata[
                                         signal_annotation.sample[i + 1] - 90:signal_annotation.sample[i + 1] + 90]
                    data.append(ventricular_signal)
        data = np.array(data)

    return data


def MITinsresamp(data):
    data_process = []
    for dat in data:
        dat = resample(dat, int(dat.shape[0]*0.6), axis=0)
        if dat.shape[1] != 2:
            dat = dat[:, 0:2]
        dat = torch.tensor(dat)
        dat = dat.transpose(0, 2)
        dat = np.array(dat)
        data_process.append(dat)
    return np.array(data_process)

#MITspra instance
def preprosessmitspra():
    for folder in ['mit-bih-supraventricular-arrhythmia-database-1.0.0/headatatr/']:
        filenames = os.listdir('C:/Users/num4/Desktop/datapre/' + folder)
        data = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('C:/Users/num4/Desktop/datapre/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('C:/Users/num4/Desktop/datapre/' + folder + name, "atr", sampfrom=0,
                                               sampto=230400)
                for i in range(60*(len(signal_annotation.sample) // 60 -1)+1):
                    ventricular_signal = ECGdata[signal_annotation.sample[i+1] - 70:signal_annotation.sample[i+1] + 70]
                    data.append(ventricular_signal)
        data = np.array(data)
    return data


def loadmitspradatas():
    b = preprosessmitspra()
    data = []
    len = b.shape[0]//60
    for i in range(217):
        datae = []
        for j in range(60*i+1, 60*(i+1)+1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data1 = data.transpose(2, 3)
    data = []
    for i in range(218, 947):
        datae = []
        for j in range(60*i+1, 60*(i+1)+1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data2 = data.transpose(2, 3)
    data = []
    for i in range(948, 1121):
        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data3 = data.transpose(2, 3)
    data = []
    for i in range(1122, 1516):
        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data4 = data.transpose(2, 3)
    data = []
    for i in range(1517, 1593):
        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data5 = data.transpose(2, 3)
    data = []
    for i in range(1594, 1733):
        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data6 = data.transpose(2, 3)
    data = []
    for i in range(1734, len-1):
        datae = []
        for j in range(60*i+1, 60*(i+1)+1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data7 = data.transpose(2, 3)
    data = torch.cat([data1, data2], dim=0)
    data = torch.cat([data, data3], dim=0)
    data = torch.cat([data, data4], dim=0)
    data = torch.cat([data, data5], dim=0)
    data = torch.cat([data, data6], dim=0)
    data = torch.cat([data, data7], dim=0)
    return data
#MITspra instance


#MITspra bag data & label(multi_hot)
def prosessmitspra():
    label_dict = {'N': 0, 'S': 1, '~': 2, '|': 3, 'V': 4, 'F': 5, 'a': 1, 'Q': 6, 'B': 0, 'J': 1}
    ecg_counter = 0
    for folder in ['mit-bih-supraventricular-arrhythmia-database-1.0.0/headatatr/']:
        filenames = os.listdir('C:/Users/num4/Desktop/datapre/' + folder)
        a = []
        lables = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('C:/Users/num4/Desktop/datapre/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('C:/Users/num4/Desktop/datapre/' + folder + name, "atr", sampfrom=0, sampto=230400)
                b = []
                for i in range(len(signal_annotation.sample) // 60 -1):
                    ventricular_signal = ECGdata[signal_annotation.sample[60 * i+1] - 50:signal_annotation.sample[                                                                               60 * (i + 1)+1] + 50]
                    ventricular_signal = ventricular_signal.astype(float)
                    beat_lable = str(signal_annotation.symbol)
                    beat_lable = beat_lable.split("', '")
                    beat_lables = beat_lable[60 * i + 1:60 * (i + 1)+1]
                    a.append(ventricular_signal)
                    b.append(beat_lables)
                b = np.array(b)
                for i in range(len(b)):
                    ecg_counter += 1
                    d = np.unique(b[i])
                    if len(d) > 1:
                        d = d[~np.isin(d, 'N')]
                        d = d[~np.isin(d, '+')]
                    elif len(d) == 1:
                        d = d
                    for i in range(len(d)):
                        d[i] = label_dict[d[i]]
                    d = list(d)
                    d = list(map(int, d))  # 使用内置map返回一个map对象，再用list将其转换为列表
                    d = np.array(d)
                    lables.append(d)
        datas = np.array(a)
        lables = np.array(lables)
    mlb = MultiLabelBinarizer(classes=[i for i in range(7)])
    y = mlb.fit_transform(lables)
    lable = np.array(y)
    datas = np.delete(datas, 217)
    lable = np.delete(lable, 217, 0)
    datas = np.delete(datas, 946)
    lable = np.delete(lable, 946, 0)
    datas = np.delete(datas, 1119)
    lable = np.delete(lable, 1119, 0)
    datas = np.delete(datas, 1513)
    lable = np.delete(lable, 1513, 0)
    datas = np.delete(datas, 1589)
    lable = np.delete(lable, 1589, 0)
    datas = np.delete(datas, 1728)
    lable = np.delete(lable, 1728, 0)

    return datas, lable
#MITspra bag data & label(multi_hot)


def STPEtranstoimage(datas):
    images = []
    for i in range(len(datas)):
        ar = torch.tensor(datas[i])
        # print(ar.shape)
        da = ar.transpose(0, 1)
        gasf = GramianAngularField(image_size=180, method='summation')
        image = gasf.fit_transform(da)
        # print(image.shape)
        images.append(image)
    images = torch.tensor(images)
    # images = images.numpy()
    return images


def MITspra():
    datas, labels = prosessmitspra()
    instances = loadmitspradatas()
    data_num = len(labels)
    shuffle_ix = np.random.permutation(np.arange(data_num))
    data = datas[shuffle_ix]
    labels = labels[shuffle_ix]
    instances = instances[shuffle_ix]

    X_train1 = instances[int(data_num * 0.3):]
    X_train2 = data[int(data_num * 0.3):]
    y_train = labels[int(data_num * 0.3):]

    X_test1 = instances[:int(data_num * 0.3)]
    X_test2 = data[:int(data_num * 0.3)]
    y_test = labels[:int(data_num * 0.3)]

    X_train1, X_test1 = preprocess_signals(X_train1, X_test1)
    X_train2, X_test2 = preprocess_signals(X_train2, X_test2)

    return X_train1, X_train2, y_train, X_test1, X_test2, y_test

