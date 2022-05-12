# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.optim.lr_scheduler import StepLR

from data_process import nn_seq_wind


def train(args, model):
    model.train()
    Dtr, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('training...')
    loss_function = nn.MSELoss().to(args.device)
    loss = 0
    for epoch in range(args.E):
        for (seq, label) in Dtr:
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_step.step()

        print('epoch', epoch, ':', loss.item())

    return model


def test(args, ann):
    ann.eval()
    Dtr, Dte = nn_seq_wind(ann.name, args.B)
    pred = []
    y = []
    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))

    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))
