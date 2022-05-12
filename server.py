# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""

import torch

from client import train, test
from model import ANN
import copy


class FedPer:
    def __init__(self, args):
        self.args = args
        self.nn = ANN(args=self.args, name='server').to(args.device)
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # dispatch
            self.dispatch()
            # local updating
            self.client_update()
            # aggregation
            self.aggregation()

        return self.nn

    def aggregation(self):
        s = 0
        for j in range(self.args.K):
            # normal
            s += self.nns[j].len

        # 基础层置零
        for v in self.nn.parameters():
            v.data.zero_()

        for j in range(self.args.K):
            cnt = 0
            for v1, v2 in zip(self.nn.parameters(), self.nns[j].parameters()):
                v1.data += v2.data * (self.nns[j].len / s)
                cnt += 1
                if cnt == 2 * (self.args.total - self.args.Kp):
                    break

    def dispatch(self):
        for j in range(self.args.K):
            cnt = 0
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()
                cnt += 1
                if cnt == 2 * (self.args.total - self.args.Kp):
                    break

    def client_update(self):  # update nn
        for k in range(self.args.K):
            self.nns[k] = train(self.args, self.nns[k])

    def global_test(self):
        for j in range(self.args.K):
            model = self.nns[j]
            model.eval()
            test(self.args, model)
