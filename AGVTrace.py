import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from scipy.special import jv
from datetime import datetime
from get_RandomPoint_InCircle import getRandomPointInCircle  #导入在园内随机取点函数
from SplitNumber import split_number # 将数随机分成n份


def AGV_trace(x):
    y = x**0.9+1/10*x
    return y

# # 限制随机分配算法
# def min_max_limit_allocate(m_people, total_money, per_max_money, per_min_money):
#     # 限制1: total_money <= m_people * max_money and total_money >= m_people * per_min_money
#     if total_money > m_people * per_max_money or total_money < m_people * per_min_money:
#         return None
#     # 1. 初步随机分配
#     allocate_list = np.random.dirichlet(np.ones(m_people)) * total_money
#     allocate_list = allocate_list.astype(int)
#     allocate_list = allocate_list.tolist()
#     # 2. 剩余随机分配
#     res = total_money - sum(allocate_list)
#     for i in range(res):
#         index = np.random.randint(m_people, size=1)[0]
#         allocate_list[index] += 1
#     # 3. 限制重分配
#     for i in range(len(allocate_list)):
#         # 限制2: everyone's money <= per_max_money
#         if allocate_list[i] > per_max_money:
#             # 记录多分的钱
#             overmuch = allocate_list[i] - per_max_money
#             # 重新分配
#             allocate_list[i] = per_max_money
#             # 把多分的钱分给其他人
#             while overmuch > 0:
#                 index = np.random.randint(m_people, size=1)[0]
#                 if allocate_list[index] < per_max_money:
#                     allocate_list[index] += 1
#                     overmuch -= 1
#         # 限制3: everyone's money >= per_min_money
#         if allocate_list[i] < per_min_money:
#             # 记录少分的钱
#             not_enough_money = per_min_money - allocate_list[i]
#             # 重新分配
#             allocate_list[i] = per_min_money
#             # 少分的钱从其他人拿
#             while not_enough_money > 0:
#                 index = np.random.randint(m_people, size=1)[0]
#                 if allocate_list[index] > per_min_money:
#                     allocate_list[index] -= 1
#                     not_enough_money -= 1
#     # 返回一个python的list作为分配结果
#     return allocate_list
#
#
# if __name__ == '__main__':
#     # 指定人数（大于0）
#     n_people = 23
#     # 奖金池奖金数(大于0)
#     money = 50
#     # 单人最小奖金设置
#     min_money = 2
#     # 单人最大奖金设置
#     max_money = 10
#     # 分配
#     allocation = min_max_limit_allocate(n_people, money, max_money, min_money)
#     # 宣布分配结果
#     print(allocation)


