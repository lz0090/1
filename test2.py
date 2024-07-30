import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from scipy.special import jv
from datetime import datetime

import test_myenv
from get_RandomPoint_InCircle import getRandomPointInCircle  #导入在园内随机取点函数
from SplitNumber import split_number # 将数随机分成n份
from AGVTrace import AGV_trace
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
from test_myenv import MyEnv
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
import ast

state_action = dict()
state_action = {
                23: [4, 3, 4, 3, 3, 3,   3, 4, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 17],
                25: [4, 3, 4, 3, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
                27: [4, 3, 4, 3, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
                29: [4, 4, 3, 4, 3, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 15],
                31: [3, 4, 3, 4, 4, 3,   4, 3, 3, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 15],
                33: [4, 4, 4, 4, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
                35: [4, 4, 3, 4, 4, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
                37: [4, 4, 3, 3, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 13],
                39: [4, 3, 4, 4, 4, 3,   4, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 13],
                41: [4, 4, 4, 4, 4, 4,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
                43: [4, 4, 3, 4, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
                45: [4, 4, 3, 4, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
                47: [4, 4, 4, 4, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 11],
                49: [4, 4, 4, 4, 4, 4,   4, 3, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 11],
                51: [3, 3, 4, 4, 4, 4,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
                53: [4, 4, 4, 4, 3, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
                55: [4, 4, 4, 4, 4, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 13],
                57: [4, 3, 4, 4, 3, 3,   4, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
                59: [4, 4, 3, 4, 4, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
                61: [3, 3, 4, 4, 4, 3,   4, 4, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 15],
                63: [4, 3, 4, 3, 3, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
                65: [4, 3, 3, 4, 3, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
                67: [3, 3, 3, 4, 4, 3,   3, 4, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 17],
                69: [3, 3, 3, 3, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 18],
                71: [3, 3, 4, 3, 3, 3,   3, 3, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 19],
                73: [3, 4, 3, 3, 3, 3,   3, 3, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 19],
                75: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 2, 3, 3, 2,     2, 2, 3, 3,   19],
                77: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 2, 3,   18],
                79: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 2, 3,   18],
                81: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 2, 3, 3, 3, 3,     2, 3, 3, 3,   17],
                83: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     2, 3, 3, 3,   16],
                85: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     2, 3, 3, 3,   16],
                87: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     2, 3, 3, 3,   16],
                89: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 2, 3, 3,   15],
                91: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     3, 3, 3, 3,   15],
                93: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     3, 3, 3, 3,   15],
                95: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 3, 3, 3,   14],
                97: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 3, 3, 3,   14],
                99: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 3, 3, 3,   14],
                101: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     2, 3, 3, 3,   15],
                103: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     2, 3, 3, 3,   15],
                105: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     2, 3, 3, 3,   15],
                107: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     3, 2, 3, 3,   16],
                109: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     3, 2, 3, 3,   16],
                111: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     3, 2, 3, 3,   16],
                113: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     2, 2, 3, 3,   17],
                115: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     2, 2, 3, 3,   17],
                117: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 2, 2, 3,     3, 3, 3, 3,   17],
                119: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 3, 2,   18],
                121: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 3, 2,   18],
                }




global a
num_psi = 0   #  当重分配决策变量的变化值达到阈值后进行的重分配次数
num_1 = 0     #  因为AGV链路时延不满足要求而进行的再分配次数
num_all = 0   #  总分配次数
done = False
for i in range(5):
    delay_alpha = []
    num_psi_alpha = []
    num_1_alpha = []
    num_all_alpha = []
    num_differ = []
    alpha = 0.1 + 0.2 * i
    for j in range(5):
        delta_psi_decision_max = 0.05 + 0.1 * j
        ever_delay = 0
        P = MyEnv()
        s = P.reset()
        num_psi = 0
        num_1 = 0
        ever_delay = 0
        num_all = 0
        done = False
        while not done:
            if s == 23:
                a = state_action[float(s)]
            s_, _, done, informa = P.step(a, alpha, delta_psi_decision_max)
            if s_ != s:
                ever_delay += (informa['T0'])

            if s_ == s:
                a = state_action[float(s_)]
                num_1 += 1

            elif test_myenv.delta_psi_decision >= delta_psi_decision_max and s_ != s and float(s_) != 123:
                a = state_action[float(s_)]
                num_psi += 1
                s = s_
            else:
                s = s_
        # print('系统平均时延:', ever_delay / 50)
        # print('重分配变量导致的重新分配:', num_psi)
        # print('AGV链路时延导致的重新分配:', num_1)
        # print('alpha:', alpha)
        # print('delta_psi_decision_max:', delta_psi_decision_max)

        delay_alpha.append(ever_delay / 50)
        num_psi_alpha.append(num_psi)
        num_1_alpha.append(num_1)
        num_all_alpha.append((num_1 + num_psi))
        num_differ.append((num_psi - num_1))

        if round(alpha,2) == 0.3 and round(delta_psi_decision_max,2) == 0.25:
            print(num_1)





    # np.save('./test2/alpha_{:.2f}.npy'.format(alpha), delay_alpha)
    # np.save('./test2/num_psi_alpha_{:.2f}.npy'.format(alpha), num_psi_alpha)
    # np.save('./test2/num_1_alpha_{:.2f}.npy'.format(alpha), num_1_alpha)
    # np.save('./test2/num_all_alpha_{:.2f}.npy'.format(alpha), num_all_alpha)
    # np.save('./test2/num_differ_{:.2f}.npy'.format(alpha), num_differ)




