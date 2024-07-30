# coding:utf-8
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
from AGVTrace import AGV_trace






' 系统环境变量 '
# ****************************************************************************************************
num_node = 2  # 边缘节点数
num_IIE_array = np.empty(num_node, dtype=int)  # 创建IIE数目数组
num_IIE_array[0] = 10  # 边缘节点1内的IIE数目
num_IIE_array[1] = 12  # 边缘节点2内的IIE数目
num_IIE = num_IIE_array.sum()                      # IIE数目的总和（不包括AGV）
coverage_node = 40  # 每个边缘节点的覆盖范围为40m
bandwith_node = 40 * np.power(10, 6)  # 每个边缘节点可分配的带宽为xxMHz （可能需要再考虑）
f_0 = 2.4 * np.power(10, 9)   # 理应是每个子载波中心频率都不一样，但中心频率远大于可分配的带宽，因此差异可以忽略不计，此外不同边缘节点中心频率应该是不一样，但考虑到差异较小，可以进行忽略
N_0 = -174      # 高斯白噪声功率谱密度为-174dBm/Hz  4 * np.float_power(10, -21)
d_0 = 1  # 路径损耗参考距离通常取1m
c_0 = 3 * np.power(10, 8) # 光速
n_loss = 1.69 # 路径损耗指数，参考ll论文生产装配车间
n_loss_agv = 4
xigema_shadow = 3.39  # 阴影衰落3.39dB，参考ll论文生产装配车间
num_slot = 50    # 时隙数 （需要再考虑）
power_tx_node = 35      # 边缘节点发射功率一样，均为35dBm
power_tx_IIE = 10       # 固定设备发射功率均为10dBm
power_tx_agv = 17       # AGV发射功率均为17dBm
num_sub_max = 50  # 每个边缘节点能够划分的最大子载波数为50
bandwith_sub = bandwith_node/num_sub_max # 每个子载波的带宽
T_IIE_max = 0.10 # 对于不动的IIE，每条链路能够允许的最大上下行时延和 0.15
T_agv_max = 0.02 # 对于AGV，其通信链路能够允许的最大上下行时延和 0.07
v_agv = 2           # AGV速度为2m/s (为x轴方向的分速度为2m/s)
data_trans = 5 * np.power(10, 6)  # 上行传输的数据量为5M
beta_trans_up = 0.01  # 设上行数据和下行数据成固定比例，即上行=下行*0.01

global h_smallfade  # 定义小尺度衰落系数
global t_allocation # 定义重分配时间
global flag_agv_tmp   # 定义AGV位置变量，取0代表AGV在节点1范围内，取1代表AGV在节点2范围内
global delta_psi_decision   # 定义决策变量的变化量为全局变量
global psi_decision_last  # 定义全局上一次重分配决策变量
global delay_agv



h_smallfade = random.gauss(0, 1)  # 定义小尺度衰落系数
t_allocation = 0 # 定义重分配时间
d_min_1 = 8.28  # AGV在节点1范围内移动时轨迹离节点1的最小值
d_min_2 = 18.79 # AGV在节点1范围内移动时轨迹离节点1的最小值
t_max_1 = 26    # 在以AGV离节点远近为判断连接原则下的AGV连接节点1的最大时刻，即在此时刻AGV要接入节点2
# alpha = 0.3   # 系统重分配时的参考变量，α越大说明越偏好子载波占用时间，越小说明偏好AGV离节点的距离
# delta_psi_decision_max = 0.25  # 重分配决策变量的变化量阈值，大于这个值代表系统需要进行重新分配

node_coordinate = np.empty([num_node, 2], dtype=int)   # 创建边缘节点的二维坐标矩阵
node_coordinate[0][0] = 43  # 边缘节点1的x坐标
node_coordinate[0][1] = 44  # 边缘节点1的y坐标
node_coordinate[1][0] = 84  # 边缘节点2的x坐标
node_coordinate[1][1] = 85  # 边缘节点2的y坐标

# IIE_coordinate_0 = getRandomPointInCircle(num_IIE_array[0], coverage_node, node_coordinate[0][0], node_coordinate[0][1])  # 创建边缘节点1内的IIE的二维坐标矩阵并随机取值
# IIE_coordinate_1 = getRandomPointInCircle(num_IIE_array[1], coverage_node, node_coordinate[1][0], node_coordinate[1][1])  # 创建边缘节点2内的IIE的二维坐标矩阵并随机取值
IIE_coordinate_0 = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13)])
IIE_coordinate_1 = np.array([(116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)])
IIE_coordinate = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13), (116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)])
agv_coordinate_init_x = 23.0 # 假定AGV初始x坐标为23 ,初始y坐标可由AGV的运动轨迹y=x^0.9+1/10*x算出

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.axis([0, 130, 0, 130])
circle1 = plt.Circle((node_coordinate[0][0], node_coordinate[0][1]), radius=coverage_node, color='c', fill=False, linestyle='--')  # 边缘节点1覆盖范围
circle2 = plt.Circle((node_coordinate[1][0], node_coordinate[1][1]), radius=coverage_node, color='m', fill=False, linestyle='--')  # 边缘节点2覆盖范围
ax.add_artist(circle1)
ax.add_artist(circle2)
plt.plot(node_coordinate[0][0], node_coordinate[0][1], 'ko')  # 对边缘节点1的坐标位置加粗显示
plt.plot(node_coordinate[1][0], node_coordinate[1][1], 'ko')  # 对边缘节点2的坐标位置加粗显示
for i in range(num_IIE_array[0]):
    plt.plot(IIE_coordinate_0[i][0], IIE_coordinate_0[i][1], 'co')   # 对边缘节点1内的IIE的坐标位置加粗显示
for i in range(num_IIE_array[1]):
    plt.plot(IIE_coordinate_1[i][0], IIE_coordinate_1[i][1], 'mo')   # 对边缘节点2内的IIE的坐标位置加粗显示

# agv_tra_x = np.linspace(0, 130)   # x取值范围
# plt.plot(agv_tra_x, AGV_trace(agv_tra_x))    # y=x^0.9+1/10*x 代表AGV的运动轨迹
# plt.show()             # 对设计的场景进行绘图

#  ****************************************************************************************************


class MyEnv(gym.Env):
    def __init__(self):
        self.viewer = None
        self.num_node = 2
        self.num_IIE_array = num_IIE_array
        self.node_coordinate = node_coordinate
        self.IIE_coordinate = IIE_coordinate
        self.agv_coordinate_init_x = agv_coordinate_init_x
        self.bandwith_node = bandwith_node
        self.coverage_node = coverage_node
        self.N_0 = N_0
        self.num_slot = num_slot
        self.power_tx_node = power_tx_node
        self.power_tx_IIE = power_tx_IIE
        self.power_tx_agv = power_tx_agv
        self.num_sub_max = num_sub_max
        self.v_agv = v_agv                         # 初始化赋值

        self.states = np.zeros([(num_slot + 1), 1], dtype=np.float32)  # 创建总的状态空间，每一个状态即为AGV的x坐标

        for t in range(num_slot):
            self.states[t] = agv_coordinate_init_x + t * v_agv  # 初始化系统状态空间，系统状态即为AGV的x坐标，num_slot取的50，即为1s取一次点，暂定这里的1s即为一个节拍点

        self.terminate_state = agv_coordinate_init_x + num_slot * v_agv  # 终止状态 ，当AGV运动至最后一个时隙时达到终止装态
        self.states[num_slot] = self.terminate_state






        self.get_t = dict()   # 创建一个AGV坐标-时隙字典，根据当前的状态（即AGV的位置），获取当前的时隙
        for t in range(num_slot):
            temp = agv_coordinate_init_x + t * v_agv
            self.get_t.update({temp: t})





        self.low = np.ones([(num_IIE+1),1], dtype=np.float32)  # 动作空间的下限，最少分配一个子载波
        self.high = 24 * np.ones([(num_IIE + 1), 1], dtype=np.float32)  # 动作空间的上限，50表示对IIE分配的子载波数为50，上下限是取不到的

        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Discrete(num_slot)




    def step(self, action, x, y):
        alpha = x
        delta_psi_decision_max = y




        global h_smallfade  # 定义小尺度衰落系数
        global t_allocation  # 定义重分配时间
        global flag_agv_tmp  # 定义AGV位置变量，取0代表AGV在节点1范围内，取1代表AGV在节点2范围内
        global psi_decision_last  # 定义全局上一次重分配决策变量
        global delta_psi_decision  # 定义决策变量的变化量为全局变量
        global flag_agv
        global delay_agv


        t_last = 0
        reward = 0  # 定义初始奖励值为0


        reward_temporary = 0 # 定义一个中间过程的奖励值，用于最后的奖励求和
        done = False  # 判断一个episode是否结束
        num_action_1 = 0  # 节点1范围内动作即子载波数的和，用于判断是否满足约束C4
        num_action_2 = 0  # 节点2范围内动作即子载波数的和，用于判断是否满足约束C4
        psi_decision = 0  # 定义重分配决策变量ψ（t）
        action_tmp1 = np.array([]) # 定义一个临时存放数组，用来存放满足flag_C2约束的action[j]值即对应的满足链路时延约束的带宽值
        action_tmp2 = np.array([])  # 定义一个临时存放数组，用来存放没有满足flag_C2约束的action[j]值即对应的满足链路时延约束的带宽值
        T_tmp1 = 0       # 定义一个临时存放数组，用来存放满足flag_C2约束的action[j]值即对应的满足链路时延约束的链路时延
        T_tmp11 = 0
        T_tmp2 = 0       # 定义一个临时存放数组，用来存放没有满足flag_C2约束的action[j]值即对应的满足链路时延约束的链路时延
        info = {0}

        flag_agv_tmp = 0
        flag_agv = 0


        gamma_down_dB = np.zeros([(num_IIE + 1), 1], dtype=np.float32)   # 创建下行链路IIE和AGV处的信噪比γ一维矩阵(dB形式)，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV
        gamma_down = np.zeros([(num_IIE + 1), 1], dtype=np.float32)      # 创建下行链路IIE和AGV处的信噪比γ一维矩阵，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV
        P_RX_down = np.zeros([(num_IIE + 1), 1], dtype=np.float32)       # 创建下行链路IIE和AGV处的接收功率一维矩阵(dB形式)，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV
        T_down = np.zeros([(num_IIE + 1), 1], dtype=np.float32)          # 创建下行链路时延一维矩阵，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV

        gamma_up_dB = np.zeros([(num_IIE + 1), 1], dtype=np.float32)     # 创建上行链路边缘节点处的信噪比γ一维矩阵(dB形式)，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV
        gamma_up = np.zeros([(num_IIE + 1), 1], dtype=np.float32)        # 创建上行链路边缘节点处的信噪比γ一维矩阵，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV
        P_RX_up = np.zeros([(num_IIE + 1), 1], dtype=np.float32)         # 创建上行链路边缘节点处的接收功率一维矩阵(dB形式)，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV
        T_up = np.zeros([(num_IIE + 1), 1], dtype=np.float32)            # 创建上行链路时延一维矩阵，前11表示边缘节点1范围内的IIE，中间13个表示结点2范围内的IIE，最后一个表示AGV

        state = self.state # 获取当前状态
        t_current = self.get_t[state[0]]  # 根据AGVx坐标-时隙字典，由当前状态-AGV的位置得到当前时隙

        epsilon_smallfade = jv(0, 2 * mt.pi * ((v_agv * f_0) / c_0) * t_current)  # AGV小尺度参数中的量化两个连续时间间隔中的信道相关性参数
        h_smallfade = epsilon_smallfade * h_smallfade + mt.sqrt((1 - epsilon_smallfade ** 2) / 2) * (random.gauss(0, 1) + (1j) * random.gauss(0, 1))  # 小尺度衰落量，复数形式
        h_smallfade_dB = 10 * mt.log(abs(h_smallfade) ** 2, 10)  # AGV的小尺度衰落量，dB形式

        flag_C2 = 0 # 优化问题中的C2约束（对不动IIE的通信链路时延的约束），即当单条链路上下行延时和小于T_IIE_max时，flag_C2取值加1 在每一个节拍即每一个不同t值时都要重置为0
        flag_C3 = 0 # 优化问题中的C3约束（对AGV通信链路时延的约束），即当AGV通信链路上下行延时和小于T_agv_max时，flag_C3值取1 在每一个节拍即每一个不同t值时都要重置为0
        flag_C4 = 0 # 优化问题中的C4约束（对边缘节点分配的子载波和的约束），即当节点通信范围内的设备的分配的子载波数和小于最大可分配子载波数num_sub_max时，flag_C4取1 在每一个节拍即每一个不同t值时都要重置为0
        flag_C5 = 0 # 优化问题中的C5约束（确保每个IIE和AGV分到的子载波数要大于0） ，全都大于0取值25， 在每一个节拍即每一个不同t值时都要重置为0

        for i in range(num_IIE + 1):  # 对所有不动IIE和AGV分配到的带宽进行非0判断，大于0flag_C5加1
            if action[i] > 0:
                flag_C5 += 1

        if flag_C5 != (num_IIE + 1) :
            next_state = state

            reward += -(num_IIE + 1 - flag_C5) * 10 * 5000

            self.state = next_state
            info = {'T': 55}  # info值返回错误原因
            return next_state, reward, done, info


##############################################
        if (mt.pow(state[0] - node_coordinate[0][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[0][1], 2)) <= (mt.pow(state[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[1][1], 2)) :
            flag_agv = 0

        if (mt.pow(state[0] - node_coordinate[0][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[0][1], 2)) > (mt.pow(state[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[1][1], 2)) and flag_agv == 0:   # 只能在flag_agv == 0时执行一次
            flag_agv = 1



#############################################

        if flag_agv == 0:  # flag_agv取0说明agv在节点1范围内
            if self.state == np.array([agv_coordinate_init_x]):
                psi_decision_last = alpha * ((t_last - t_allocation) / (t_max_1 - 0)) + (1 - alpha) * ((mt.sqrt(mt.pow(agv_coordinate_init_x - node_coordinate[0][0], 2) + mt.pow(AGV_trace(agv_coordinate_init_x) - node_coordinate[0][1], 2)) - d_min_1) / (coverage_node - d_min_1))  # 当AGV位于初始状态时决策变量值即为上一次
            "求下行链路时延"
            for i in range(num_IIE_array[0]):
                P_RX_down[i] = power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[0][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[0][1], 2)) / d_0, 10) + xigema_shadow)               # 求节点1范围内IIE的dB形式的接收功率


            for i in range(num_IIE_array[0], num_IIE):
                P_RX_down[i] = power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[1][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[1][1], 2)) / d_0, 10) + xigema_shadow)               # 求节点2范围内IIE的dB形式的接收功率

            P_RX_down[num_IIE] = power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss_agv * mt.log(mt.sqrt(mt.pow(abs(state[0] - node_coordinate[0][0]), 2.6) + mt.pow(abs(AGV_trace(state[0]) - node_coordinate[0][1]), 2.6)) / d_0, 10) + xigema_shadow)       # 求节点1范围内AGV的dB形式的接收功率

            for i in range(num_IIE_array[0]):
                gamma_down_dB[i] = P_RX_down[i]-(N_0 + 10 *  mt.log(bandwith_sub * mt.ceil(action[i]), 10) + power_tx_node-(20 * mt.log(((4 * mt.pi * f_0 * d_0)/c_0), 10) +\
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[1][0], 2)+mt.pow(IIE_coordinate[i][1] - node_coordinate[1][1], 2))/d_0, 10) + xigema_shadow))                  # 求节点1范围内IIE的dB形式的信噪比

            for i in range(num_IIE_array[0], num_IIE):
                gamma_down_dB[i] = P_RX_down[i]-(N_0 + 10 *  mt.log(bandwith_sub * mt.ceil(action[i]), 10) + power_tx_node-(20 * mt.log(((4 * mt.pi * f_0 * d_0)/c_0), 10) +\
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[0][0], 2)+mt.pow(IIE_coordinate[i][1] - node_coordinate[0][1], 2))/d_0, 10) + xigema_shadow))                  # 求结点2范围内IIE的dB形式的信噪比

            gamma_down_dB[num_IIE] = P_RX_down[num_IIE]-((N_0 + 10 *  mt.log(bandwith_sub * mt.ceil(action[num_IIE]), 10)) + (power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0)/c_0), 10) +\
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(state[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[1][1], 2))/d_0, 10) + xigema_shadow) ))          # 求结点1范围内AGV的dB形式的信噪比

            for i in range(num_IIE + 1):
                gamma_down[i] = 10 ** (gamma_down_dB[i] / 10)  # 将节点1和2范围内dB形式的信噪比变成普通形式


            for i in range(num_IIE ):
                T_down[i] = data_trans / (bandwith_sub * mt.ceil(action[i]) * mt.log(1 + gamma_down[i], 2))  # 求出边缘节点1和2范围内的下行链路时延
            T_down[num_IIE] = data_trans / (bandwith_sub * mt.ceil(action[num_IIE]) * mt.log(1 + gamma_down[num_IIE], 2))
            # print(T_down[num_IIE])




            "求上行链路时延"
            for i in range(num_IIE_array[0]):
                P_RX_up[i] = power_tx_IIE - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[0][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[0][1], 2)) / d_0, 10) + xigema_shadow)  # 求节点1范围内dB形式的上行链路接收功率

            for i in range(num_IIE_array[0], num_IIE):
                P_RX_up[i] = power_tx_IIE - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                          10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[1][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[1][1], 2)) / d_0, 10) + xigema_shadow)    # 求节点2范围内dB形式的上行链路接收功率

            P_RX_up[num_IIE] = power_tx_agv - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(abs(state[0] - node_coordinate[0][0]), 2) + mt.pow(abs(AGV_trace(state[0]) - node_coordinate[0][1]), 2)) / d_0, 10) + xigema_shadow)      # 求节点1范围内AGV的dB形式的上行链路接收功率

            for i in range(num_IIE + 1):
                gamma_up_dB[i] = P_RX_up[i] - (N_0 + 10 * mt.log(bandwith_sub * mt.ceil(action[i]), 10))  # 求dB形式的上行链路信噪比

            for i in range(num_IIE + 1):
                gamma_up[i] = 10 ** (gamma_up_dB[i] / 10)  # 将dB形式的上行链路信噪比变成普通形式

            for i in range(num_IIE ):
                T_up[i] = (beta_trans_up * data_trans) / (bandwith_sub * mt.ceil(action[i]) * mt.log(1 + gamma_up[i], 2))  # 求出边缘节点1和2范围内的上行链路时延
            T_up[num_IIE] = (beta_trans_up * data_trans) / (bandwith_sub * mt.ceil(action[num_IIE]) * mt.log(1 + gamma_up[num_IIE], 2))
            delay_agv = T_down[num_IIE] + T_up[num_IIE]



            for i in range(num_IIE_array[0]):
                T_tmp1 += (T_up[i][0] + T_down[i][0])






            for i in range(num_IIE):  # 对每一条IIE通信链路上下行链路时延和进行判断，小于最大值flag_C2加1
                if T_down[i] + T_up[i] <= T_IIE_max :
                    action_tmp1 = np.append(action_tmp1, action[i])  # 满足单条链路的时延约束的存放在action_tmp1
                    flag_C2 += 1

                else:
                    action_tmp2 = np.append(action_tmp2, action[i])  # 不满足单条链路的时延约束的存放在action_tmp2


            if T_down[num_IIE] + T_up[num_IIE] <= T_agv_max : # 对AGV通信链路上下行链路时延和进行判断，小于最大值flag_C3取1
                flag_C3 = 1

            for i in range(num_IIE_array[0]):  # 求节点1范围内分配的子载波数和
                num_action_1 += mt.ceil(action[i])

            for i in range(num_IIE_array[0], num_IIE):  # 求结点2范围内分配的子载波数和
                num_action_2 += mt.ceil(action[i])
            if num_action_1 <= (num_sub_max - mt.ceil(action[num_IIE]))  :
                flag_C4 = 1

########################################################################################################################
            if flag_C3 != 1:
                next_state = state
                reward = - ((T_up[num_IIE][0] + T_down[num_IIE][0]) / T_agv_max) * 70 * 50 * (1 - flag_C3)
                for i in range(num_IIE_array[0]):
                    reward += - max(mt.ceil(action[i]) - 2, 0) * 300
                # for i in range(num_IIE_array[0], num_IIE):
                #     reward += - max(mt.ceil(action[i]) -4 , 0) * 50
                self.state = next_state
                info = {'T': ((0.6 + 0.6 + (T_up[num_IIE][0] + T_down[num_IIE][0]) - T_agv_max) / (num_IIE + 1) * 1000)}  # info值返回错误原因
                return next_state, reward, done, info
########################################################################################################################


########################################################################################################################
            if flag_C4 != 1 and flag_C3 == 1:
                next_state = state
                # reward = -(max(num_action_1 - (num_sub_max - mt.ceil(action[num_IIE])), 0) * 5 + max(num_action_2 - num_sub_max, 0) * 5) * (T_agv_max / (T_up[num_IIE][0] + T_down[num_IIE][0]))
                for i in range(num_IIE_array[0]):
                    reward += - max(mt.ceil(action[i]) -2 , 0) * 120
                # for i in range(num_IIE_array[0], num_IIE):
                #     reward += - max(mt.ceil(action[i]) -4 , 0) * 70
                # reward = reward * (T_agv_max / (T_up[num_IIE][0] + T_down[num_IIE][0]))


                self.state = next_state
                # info = {'T': [0.5, next_state,flag_C2,flag_C4,num_action_1 , num_action_2]}
                info = {'T': ((0.55 * (num_action_1 / (num_sub_max - mt.ceil(action[num_IIE]))) + 0.6) / (num_IIE + 1) * 1000)}
                return next_state, reward, done, info
########################################################################################################################

            if flag_C2 == num_IIE and flag_C3 == 1 and flag_C4 == 1  :  # 判断约束C2，C3，C4和C5是否同时满足

                for i in range(num_IIE_array[0]):
                    T_tmp11 += (T_up[i][0] + T_down[i][0])
                reward = (num_IIE * num_IIE_array[0]) / (T_tmp11)  * 5

                reward = reward * (1 / (T_agv_max - (T_up[num_IIE][0] + T_down[num_IIE][0])))
                next_state = self.states[t_current + 1]
                # print(state[0], ':', (T_down.sum() + T_up.sum()) / (num_IIE + 1))

                # info  = {'T' : (T_tmp11 + 0.49470636) / (num_IIE + 1) * 1000} # info用于返回满足所有约束时系统的平均时延
                info = {'T0': (T_down.sum() + T_up.sum()) / (num_IIE + 1) * 1000}



            else: # 约束条件都没有满足，系统需要继续在此状态

                for i in range(num_IIE_array[0]):
                    reward += max((action[i])  / 3 , 1) * 0.1

                for i in range(num_IIE_array[0], num_IIE):
                    reward += max((action[i])  / 4 , 1) * 0.1
                reward = reward * (((T_up[num_IIE][0] + T_down[num_IIE][0]) / T_agv_max))**2



                next_state = state
                # info = {'T': [next_state,flag_C2,flag_C3,flag_C4,flag_C5]}
                info = {'T': ((T_tmp1 + 0.6) / (num_IIE + 1)) * 1000}

            self.state = next_state

            if next_state == self.terminate_state :
                done = True

            state_tmp1 = self.state  # 获取下一状态便于计算flag_agv_tmp
            t_current_tmp1 = self.get_t[state_tmp1[0]]  # 根据AGVx坐标-时隙字典，由下一状态-AGV的位置得到下一时隙

            psi_decision_current = alpha * ((t_current_tmp1 - t_allocation) / (t_max_1 - 0)) + (1 - alpha) * ((mt.sqrt(mt.pow(state_tmp1[0] - node_coordinate[0][0], 2) + mt.pow(AGV_trace(state_tmp1[0]) - node_coordinate[0][1],2)) - d_min_1) / (coverage_node - d_min_1))  # 满足全部约束情况下计算下一时刻的重分配决策值
            delta_psi_decision = abs(psi_decision_current - psi_decision_last)
            if delta_psi_decision >= delta_psi_decision_max:
                # psi_decision_last = psi_decision_current
                t_allocation = t_current_tmp1
                psi_decision_last = alpha * ((t_current_tmp1 - t_allocation) / (t_max_1 - 0)) + (1 - alpha) * ((mt.sqrt(mt.pow(state_tmp1[0] - node_coordinate[0][0], 2) + mt.pow(AGV_trace(state_tmp1[0]) - node_coordinate[0][1],2)) - d_min_1) / (coverage_node - d_min_1))


            if (mt.pow(state_tmp1[0] - node_coordinate[0][0], 2) + mt.pow(AGV_trace(state_tmp1[0]) - node_coordinate[0][1], 2)) > (mt.pow(state_tmp1[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state_tmp1[0]) - node_coordinate[1][1], 2)) and info != {'action is not OK': 0}:
                flag_agv_tmp = 1
            else:
                flag_agv_tmp = 0



        if flag_agv == 1:  # flag_agv取1说明agv在节点2范围内
            if self.state == np.array([75]):
                t_allocation = t_current  # AGV切换到节点2，之前在节点1处累计的t_allocation即子载波占用时间要变为当前时刻，即切换时刻
                t_last = 26
                psi_decision_last = alpha * ((t_last - t_allocation) / (num_slot - t_max_1)) + (1 - alpha) * ((mt.sqrt(mt.pow(state[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[1][1], 2)) - d_min_2) / (coverage_node - d_min_2))  # 上一次的决策变量值就是切换时的决策变量值



            "求下行链路时延"
            for i in range(num_IIE_array[0]):
                P_RX_down[i] = power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[0][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[0][1], 2)) / d_0, 10) + xigema_shadow)               # 求节点1范围内IIE的dB形式的接收功率

            for i in range(num_IIE_array[0], num_IIE):
                P_RX_down[i] = power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[1][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[1][1], 2)) / d_0, 10) + xigema_shadow)               # 求节点2范围内IIE的dB形式的接收功率

            P_RX_down[num_IIE] = power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss_agv * mt.log(mt.sqrt(mt.pow(abs(state[0] - node_coordinate[1][0]), 2.6) + mt.pow(abs(AGV_trace(state[0]) - node_coordinate[1][1]), 2.6)) / d_0, 10) + xigema_shadow)       # 求节点2范围内AGV的dB形式的接收功率

            for i in range(num_IIE_array[0]):
                gamma_down_dB[i] = P_RX_down[i]-(N_0 + 10 *  mt.log(bandwith_sub * mt.ceil(action[i]), 10) + power_tx_node-(20 * mt.log(((4 * mt.pi * f_0 * d_0)/c_0), 10) +\
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[1][0], 2)+mt.pow(IIE_coordinate[i][1] - node_coordinate[1][1], 2))/d_0, 10) + xigema_shadow))                  # 求节点1范围内IIE的dB形式的信噪比

            for i in range(num_IIE_array[0], num_IIE):
                gamma_down_dB[i] = P_RX_down[i]-(N_0 + 10 *  mt.log(bandwith_sub * mt.ceil(action[i]), 10) + power_tx_node-(20 * mt.log(((4 * mt.pi * f_0 * d_0)/c_0), 10) +\
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[0][0], 2)+mt.pow(IIE_coordinate[i][1] - node_coordinate[0][1], 2))/d_0, 10) + xigema_shadow))                  # 求结点2范围内IIE的dB形式的信噪比

            gamma_down_dB[num_IIE] = P_RX_down[num_IIE]-((N_0 + 10 *  mt.log(bandwith_sub * mt.ceil(action[num_IIE]), 10)) + (power_tx_node - (20 * mt.log(((4 * mt.pi * f_0 * d_0)/c_0), 10) +\
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(state[0] - node_coordinate[0][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[0][1], 2))/d_0, 10) + xigema_shadow) ))          # 求结点1范围内AGV的dB形式的信噪比

            for i in range(num_IIE + 1):
                gamma_down[i] = 10 ** (gamma_down_dB[i] / 10)  # 将节点1和2范围内dB形式的信噪比变成普通形式

            for i in range(num_IIE ):
                T_down[i] = data_trans / (bandwith_sub * mt.ceil(action[i]) * mt.log(1 + gamma_down[i], 2))  # 求出边缘节点1和2范围内的下行链路时延
            T_down[num_IIE] = data_trans / (bandwith_sub * mt.ceil(action[num_IIE]) * mt.log(1 + gamma_down[num_IIE], 2))






            "求上行链路时延"
            for i in range(num_IIE_array[0]):
                P_RX_up[i] = power_tx_IIE - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[0][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[0][1], 2)) / d_0, 10) + xigema_shadow)  # 求节点1范围内dB形式的上行链路接收功率

            for i in range(num_IIE_array[0], num_IIE):
                P_RX_up[i] = power_tx_IIE - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                          10 * n_loss * mt.log(mt.sqrt(mt.pow(IIE_coordinate[i][0] - node_coordinate[1][0], 2) + mt.pow(IIE_coordinate[i][1] - node_coordinate[1][1], 2)) / d_0, 10) + xigema_shadow)    # 求节点2范围内dB形式的上行链路接收功率

            P_RX_up[num_IIE] = power_tx_agv - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                            10 * n_loss * mt.log(mt.sqrt(mt.pow(state[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state[0]) - node_coordinate[1][1], 2)) / d_0, 10) + xigema_shadow)       # 求节点1范围内AGV的dB形式的上行链路接收功率

            for i in range(num_IIE + 1):
                gamma_up_dB[i] = P_RX_up[i] - (N_0 + 10 * mt.log(bandwith_sub * mt.ceil(action[i]), 10))  # 求dB形式的上行链路信噪比

            for i in range(num_IIE + 1):
                gamma_up[i] = 10 ** (gamma_up_dB[i] / 10)  # 将dB形式的上行链路信噪比变成普通形式

            for i in range(num_IIE + 1):
                T_up[i] = (beta_trans_up * data_trans) / (bandwith_sub * mt.ceil(action[i]) * mt.log(1 + gamma_up[i], 2))  # 求出边缘节点1和2范围内的上行链路时延
            T_up[num_IIE] = (beta_trans_up * data_trans) / (bandwith_sub * mt.ceil(action[num_IIE]) * mt.log(1 + gamma_up[num_IIE], 2))





            for i in range(num_IIE):  # 对每一条IIE通信链路上下行链路时延和进行判断，小于最大值flag_C2加1
                if T_down[i] + T_up[i] <= T_IIE_max :
                    action_tmp1 = np.append(action_tmp1, action[i])  # 满足单条链路的时延约束的存放在action_tmp1
                    flag_C2 += 1

                else:
                    action_tmp2 = np.append(action_tmp2, action[i])  # 不满足单条链路的时延约束的存放在action_tmp2


            if T_down[num_IIE] + T_up[num_IIE] <= T_agv_max : # 对AGV通信链路上下行链路时延和进行判断，小于最大值flag_C3取1
                flag_C3 = 1

            for i in range(num_IIE_array[0]):  # 求节点1范围内分配的子载波数和
                num_action_1 += mt.ceil(action[i])

            for i in range(num_IIE_array[0], num_IIE):  # 求结点2范围内分配的子载波数和
                num_action_2 += mt.ceil(action[i])
            if num_action_2 <= (num_sub_max - mt.ceil(action[num_IIE]))  :
                flag_C4 = 1
########################################################################################################################
            if flag_C3 != 1:  #
                next_state = state
                reward = - ((T_up[num_IIE][0] + T_down[num_IIE][0]) / T_agv_max) * 70 * 50 * (1 - flag_C3)

                for i in range(num_IIE_array[0], num_IIE):
                    reward += - max(mt.ceil(action[i]) -1 , 0) * 500
                self.state = next_state
                info = {'T': 0.20}  # info值返回错误原因
                return next_state, reward, done, info
########################################################################################################################



########################################################################################################################
            if flag_C4 != 1 and flag_C3 == 1:
                next_state = state
                # reward = -(max(num_action_1 - num_sub_max, 0) * 5 + max(num_action_2 - (num_sub_max - mt.ceil(action[num_IIE])), 0) * 5) * (T_agv_max / (T_up[num_IIE][0] + T_down[num_IIE][0]))

                for i in range(num_IIE_array[0], num_IIE):
                    reward += - max(mt.ceil(action[i]) -1 , 0) * 500

                self.state = next_state
                info = {'T': [0.5, next_state,flag_C2,flag_C4,num_action_1 , num_action_2]}
                return next_state, reward, done, info
########################################################################################################################

            if flag_C2 == num_IIE  and flag_C4 == 1 and flag_C3 == 1:  # 判断约束C2，C3，C4和C5是否同时满足  #


                for i in range(num_IIE_array[0], num_IIE):
                    T_tmp2 += (T_up[i][0] + T_down[i][0])

                reward = (num_IIE * (num_IIE - num_IIE_array[0])) / (T_tmp2)  * 5
                reward = reward * (1 / (T_agv_max - (T_up[num_IIE][0] + T_down[num_IIE][0])))
                next_state = self.states[t_current + 1]
                # print(state[0], ':', (T_down.sum() + T_up.sum()) / (num_IIE + 1))

                info  = {'T0' : (T_down.sum() + T_up.sum()) / (num_IIE + 1) * 1000} # info用于返回满足所有约束时系统的平均时延


            else: # 约束条件都没有满足，系统需要继续在此状态


                #
                # for i in range(num_IIE_array[0]):
                #     reward += max((action[i])  / 4 , 1) * 0.1

                for i in range(num_IIE_array[0], num_IIE):
                    reward += max((action[i])  / 1 , 1) * 0.1
                reward = reward * (((T_up[num_IIE][0] + T_down[num_IIE][0]) / T_agv_max))**2




                next_state = state
                info = {'T': [next_state,flag_C2,flag_C3,flag_C4,flag_C5]}


            self.state = next_state
            if next_state == self.terminate_state :
                # print("AGV has run to end")
                done = True

            if not done:
                state_tmp2 = self.state  # 获取下一状态便于计算flag_agv_tmp
                t_current_tmp2 = self.get_t[state_tmp2[0]]  # 根据AGVx坐标-时隙字典，由下一状态-AGV的位置得到下一时隙

                psi_decision_current = alpha * ((t_current_tmp2 - t_allocation) / (num_slot - t_max_1)) + (1 - alpha) * ((mt.sqrt(mt.pow(state_tmp2[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state_tmp2[0]) - node_coordinate[1][1],2)) - d_min_2) / (coverage_node - d_min_2)) # 满足全部约束情况下计算下一时刻的重分配决策值
                delta_psi_decision = abs(psi_decision_current - psi_decision_last)
                if delta_psi_decision >= delta_psi_decision_max:
                    # psi_decision_last = psi_decision_current
                    t_allocation = t_current_tmp2
                    psi_decision_last = alpha * ((t_current_tmp2 - t_allocation) / (num_slot - t_max_1)) + (1 - alpha) * ((mt.sqrt(mt.pow(state_tmp2[0] - node_coordinate[1][0], 2) + mt.pow(AGV_trace(state_tmp2[0]) - node_coordinate[1][1],2)) - d_min_2) / (coverage_node - d_min_2))

        return next_state, reward, done, info














    def reset(self):
        self.state = np.array([agv_coordinate_init_x])
        flag_agv = 0
        flag_agv_tmp = 0

        return self.state

    def render(self, mode="human"):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


















