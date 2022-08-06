# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:22
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import sys

import numpy as np
import pandas as pd
import torch
from args import args_parser

sys.path.append('../')
from torch.utils.data import Dataset, DataLoader

args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


def load_data(file_name):
    df = pd.read_csv('data/Wind/Task 1/Task1_W_Zone1_10/' + file_name + '.csv', encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B):
    data = load_data(file_name)
    columns = data.columns
    # wind = data[columns[2]]

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[2]]), np.min(train[train.columns[2]])

    def process(dataset, batch_size, shuffle):
        wind = dataset[dataset.columns[2]]
        wind = (wind - n) / (m - n)
        wind = wind.tolist()
        dataset = dataset.values.tolist()
        seq = []
        for i in range(len(dataset) - 24):
            train_seq = []
            train_label = []

            for j in range(i, i + 24):
                x = [wind[j]]
                for c in range(3, 7):
                    x.append(dataset[j][c])
                train_seq.append(x)
            train_label.append(wind[i+24])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, shuffle=True)
    Val = process(val, B, shuffle=True)
    Dte = process(test, B, shuffle=False)

    return Dtr, Val, Dte, m, n


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))
