# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
import copy
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_process import nn_seq_wind


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss()
    val_loss = []
    for (seq, label) in Val:
        seq, label = seq.to(args.device), label.to(args.device)
        y_pred = model(seq)
        loss = loss_function(y_pred, label)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train(args, model):
    model.train()
    Dtr, Val, Dte, m, n = nn_seq_wind(model.name, args.B)
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
    best_model = None
    min_val_loss = 5
    min_epochs = 5
    for epoch in tqdm(range(args.E)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
        lr_step.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        model.train()
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        tqdm.write('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

    return best_model


def test(args, ann):
    ann.eval()
    Dtr, Val, Dte, m, n = nn_seq_wind(ann.name, args.B)
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

    y = y * (m - n) + n
    pred = pred * (m - n) + n
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))
