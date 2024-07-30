# import numpy as np
# import math as mt
# import random
# import matplotlib.pyplot as plt
# import gym
# from gym import spaces
# from scipy.special import jv
# from datetime import datetime
#
# import test_myenv
# from get_RandomPoint_InCircle import getRandomPointInCircle  #导入在园内随机取点函数
# from SplitNumber import split_number # 将数随机分成n份
# from AGVTrace import AGV_trace
# import torch
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import gym
# from test_myenv import MyEnv
# import argparse
# from normalization import Normalization, RewardScaling
# from replaybuffer import ReplayBuffer
# import ast
# from scipy import io
# state_action = dict()
# state_action = {
#                 23: [4, 3, 4, 3, 3, 3,   3, 4, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 17],
#                 25: [4, 3, 4, 3, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
#                 27: [4, 3, 4, 3, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
#                 29: [4, 4, 3, 4, 3, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 15],
#                 31: [3, 4, 3, 4, 4, 3,   4, 3, 3, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 15],
#                 33: [4, 4, 4, 4, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
#                 35: [4, 4, 3, 4, 4, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
#                 37: [4, 4, 3, 3, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 13],
#                 39: [4, 3, 4, 4, 4, 3,   4, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 13],
#                 41: [4, 4, 4, 4, 4, 4,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
#                 43: [4, 4, 3, 4, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
#                 45: [4, 4, 3, 4, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
#                 47: [4, 4, 4, 4, 4, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 11],
#                 49: [4, 4, 4, 4, 4, 4,   4, 3, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 11],
#                 51: [3, 3, 4, 4, 4, 4,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
#                 53: [4, 4, 4, 4, 3, 3,   4, 4, 4, 4,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 12],
#                 55: [4, 4, 4, 4, 4, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 13],
#                 57: [4, 3, 4, 4, 3, 3,   4, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
#                 59: [4, 4, 3, 4, 4, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 14],
#                 61: [3, 3, 4, 4, 4, 3,   4, 4, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 15],
#                 63: [4, 3, 4, 3, 3, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
#                 65: [4, 3, 3, 4, 3, 3,   4, 3, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 16],
#                 67: [3, 3, 3, 4, 4, 3,   3, 4, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 17],
#                 69: [3, 3, 3, 3, 3, 3,   3, 4, 4, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 18],
#                 71: [3, 3, 4, 3, 3, 3,   3, 3, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 19],
#                 73: [3, 4, 3, 3, 3, 3,   3, 3, 3, 3,       4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 19],
#                 75: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 2, 3, 3, 2,     2, 2, 3, 3,   19],
#                 77: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 2, 3,   18],
#                 79: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 2, 3,   18],
#                 81: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 2, 3, 3, 3, 3,     2, 3, 3, 3,   17],
#                 83: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     2, 3, 3, 3,   16],
#                 85: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     2, 3, 3, 3,   16],
#                 87: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     2, 3, 3, 3,   16],
#                 89: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 2, 3, 3,   15],
#                 91: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     3, 3, 3, 3,   15],
#                 93: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     3, 3, 3, 3,   15],
#                 95: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 3, 3, 3,   14],
#                 97: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 3, 3, 3,   14],
#                 99: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     3, 3, 3, 3,   14],
#                 101: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     2, 3, 3, 3,   15],
#                 103: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     2, 3, 3, 3,   15],
#                 105: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 3, 3,     2, 3, 3, 3,   15],
#                 107: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     3, 2, 3, 3,   16],
#                 109: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     3, 2, 3, 3,   16],
#                 111: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 2, 3, 3,     3, 2, 3, 3,   16],
#                 113: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     2, 2, 3, 3,   17],
#                 115: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 3, 3, 3,     2, 2, 3, 3,   17],
#                 117: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 2, 2, 2, 3,     3, 3, 3, 3,   17],
#                 119: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 3, 2,   18],
#                 121: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   2, 3, 3, 3, 3, 3,     2, 2, 3, 2,   18],
#                 }
#
#
#
#
# global a
# num_psi = 0   #  当重分配决策变量的变化值达到阈值后进行的重分配次数
# num_1 = 0     #  因为AGV链路时延不满足要求而进行的再分配次数
# num_all = 0   #  总分配次数
# done = False
# alpha = 0.3
# delta_psi_decision_max = 0.25
# P = MyEnv()
# s = P.reset()
# ever_delay = []
# while not done:
#     if s == 23:
#         a = state_action[float(s)]
#     s_, _, done, informa = P.step(a, alpha, delta_psi_decision_max)
#     if s_ != s:
#         ever_delay.append((informa['T0']))
#         print(s, ':', (informa['T0']))
#     if s_ == s:
#         a = state_action[float(s_)]
#         num_1 += 1
#         print('num_1:', s)
#
#     elif test_myenv.delta_psi_decision >= delta_psi_decision_max and s_ != s and float(s_) != 123:
#         a = state_action[float(s_)]
#         num_psi += 1
#         s = s_
#         print('num_psi:', s)
#     else:
#         s = s_
#
#
#
#
# # np.save('./test5/ever_delay.npy', ever_delay)
# #
# #
# # a = np.load('E:/project/PPO_static/test5/ever_delay.npy', allow_pickle=True)
# # io.savemat('E:/project/PPO_static/test5/ever_delay.mat', {'ever_delay': a})
#
#

# def max_sum_non_adjacent(numbers):
#     n = len(numbers)
#
#     if n == 0:
#         return 0
#     elif n == 1:
#         return numbers[0]
#
#     # 初始化动态规划数组
#     dp = [0] * n
#     dp[0] = numbers[0]
#     dp[1] = max(numbers[0], numbers[1])
#
#     for i in range(2, n):
#         # 状态转移方程
#         dp[i] = max(dp[i - 1], dp[i - 2] + numbers[i])
#
#     return max(dp[-1], dp[-2])
#
# # 输入一系列数字，以空格隔开
# input_numbers = input("请输入一系列数字，以空格隔开: ")
# numbers = list(map(int, input_numbers.split()))
# result = max_sum_non_adjacent(numbers)
# print("选取的数字使得和最大为:", result)

# def max_sum_non_adjacent(numbers):
#     n = len(numbers)
#
#     if n == 0:
#         return 0
#     elif n == 1:
#         return numbers[0]
#     elif n == 2:
#         return max(numbers[0], numbers[1])
#
#     # 初始化动态规划数组
#     dp = [0] * n
#     dp[0] = numbers[0]
#     dp[1] = max(numbers[0], numbers[1])
#
#     for i in range(2, n):
#         # 状态转移方程
#         dp[i] = max(dp[i - 1], dp[i - 2] + numbers[i])
#
#     return max(dp[-1], dp[-2])
#
# # 输入一系列数字，以空格隔开
# input_numbers = input("请输入一系列数字，以空格隔开: ")
# numbers = list(map(int, input_numbers.split()))
# result = max_sum_non_adjacent(numbers[1:-1])  # 排除第一个和最后一个数字

# def max_sum_non_adjacent(numbers):
#     n = len(numbers)
#
#     if n == 0:
#         return 0
#     elif n == 1:
#         return numbers[0]
#
#     # 初始化动态规划数组
#     dp = [0] * n
#     dp[0] = numbers[0]
#     dp[1] = max(numbers[0], numbers[1])
#
#     for i in range(2, n):
#         # 状态转移方程
#         dp[i] = max(dp[i - 1], dp[i - 2] + numbers[i])
#
#     return max(dp[-1], dp[-2])
#
# # 输入一系列数字，以空格隔开
# input_numbers = input("请输入一系列数字，以空格隔开: ")
# numbers = list(map(int, input_numbers.split()))
# result = max_sum_non_adjacent(numbers)
# print(numbers)
# print("选取的数字使得和最大为:", result)
# def max_sum_non_adjacent(numbers):
#     n = len(numbers)
#
#     if n == 0:
#         return 0
#     elif n == 1:
#         return numbers[0]
#
#
#     # 初始化动态规划数组
#     dp = [0] * n
#     dp[0] = numbers[0]
#     dp[1] = max(numbers[0], numbers[1])
#
#     for i in range(2, n):
#         # 状态转移方程，考虑不能同时包括第一个和最后一个数字
#         dp[i] = max(dp[i - 1], dp[i - 2] + numbers[i])
#
#     return max(dp[-1], dp[-2])
#
# # 输入一系列数字，以空格隔开
# input_numbers = input("请输入一系列数字，以空格隔开: ")
# numbers = list(map(int, input_numbers.split()))
# result = max(max_sum_non_adjacent(numbers[1:]), max_sum_non_adjacent(numbers[:-1]))
# print("选取的数字使得和最大为:", result)


def two_sum(nums, target):
    num_dict = {}  # 创建一个字典来存储元素和其索引的映射

    for i, num in enumerate(nums):
        complement = target - num

        if complement in num_dict:
            return [num_dict[complement], i]

        num_dict[num] = i

    return None  # 如果没有解决方案，返回None


# 从键盘输入整数数组
nums = map(int, input("请输入整数数组，以空格分隔: ").split())

# 从键盘输入目标值
target = int(input("请输入目标值: "))

result = two_sum(nums, target)
print(result)


