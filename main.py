# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 13:01
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from server import FedPer


def main():
    args = args_parser()
    fedPer = FedPer(args)
    fedPer.server()
    fedPer.global_test()


if __name__ == '__main__':
    main()
