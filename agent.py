# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:36 2020

@author: liangyu

Create the agent for a UE
"""

import copy
import numpy as np
from numpy import pi
from collections import namedtuple
from random import random, uniform, choice, randrange, sample
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from scenario import Scenario, BS

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple
# 一个数组 经验区的经验数逐渐增加直到满
class ReplayMemory(object):    # Define a replay memory

    def __init__(self, capacity):
        self.capacity = capacity
        # 采样时和经验排列顺序无关 没必要选择队列 数组就可以
        self.memory = []
        # 接下来接收的经验在经验池中的索引
        self.position = 0

    def Push(self, *args):
        # 实际经验数小于规定容量时 填充一个空经验占位
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # 新进的经验覆盖旧的经验或是空数据
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    #  random库中的一个函数，population为总样本，k为需要的样本数 count为重复的样本数 返回一个列表
    # def sample(self, population, k, *, counts=None)
    def Sample(self, batch_size):
        return sample(self.memory, batch_size)
    # 获取目前经验区内存储的数量 特殊方法
    def __len__(self):
        return len(self.memory)
# 继承神经网络模块中的基类nn.Module，以结构化的方式定义网络、可以使用BP优化网络中的权重、设置模块中的参数、继承有用的方法
# 设计的Q网络计算Q值
class DNN(nn.Module):  # Define a deep neural network
    #opt: 一个包含配置和选项的对象，如学习速率、容量等
    # sce: 可能是一个场景或环境的对象。
    # scenario: 与场景相关的信息或函数
    def __init__(self, opt, sce, scenario):  # Define the layers of the fully-connected hidden network
        super(DNN, self).__init__()
        # 输入层是一个线性层，它将输入的状态（可能是一个多agent的状态向量）从 opt.nagents 维转换到 64 维
        self.input_layer = nn.Linear(opt.nagents, 64)
        # 这两个都是隐藏层，它们都是线性层，用于在输入和输出之间进行非线性变换。它们分别将64维和32维的向量转换回32维
        self.middle1_layer = nn.Linear(64, 32)
        self.middle2_layer = nn.Linear(32, 32)
        # 输出层是一个线性层，它将32维的向量转换为一个广播信号，该信号的维度是场景的广播信号数量与通道数量的乘积
        self.output_layer = nn.Linear(32, scenario.BS_Number() * sce.nChannel)
	# 	这个方法定义了数据在网络中的前向传播过程
    def forward(self, state):  # Define the neural network forward function
        # 输入数据通过输入层，经过ReLU激活函数
        x1 = F.relu(self.input_layer(state))
        # x1 通过第一个隐藏层，再经过ReLU激活函数
        x2 = F.relu(self.middle1_layer(x1))
        # X2 通过第二个隐藏层，再经过ReLU激活函数
        x3 = F.relu(self.middle2_layer(x2))
        # x3 通过输出层，不经过激活函数（因为这是一个线性层）
        out = self.output_layer(x3)
        # 返回处理后的数据 out
        return out

class Agent:  # Define the agent (UE)
    # opt: 一个包含配置和选项的对象，如学习速率、容量等
    # sce: 可能是一个场景或环境的对象
    # scenario: 与场景相关的信息或函数
    # index: 基站在集合中的索引
    # device: 用于指定计算设备，例如CPU或GPU
    def __init__(self, opt, sce, scenario, index, device):  # Initialize the agent (UE)
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        # 根据scenario设置智能体/代理的位置
        self.location = self.Set_Location(scenario)
        # 初始化一个回放内存（Replay Memory）对象，容量由opt.capacity指定
        self.memory = ReplayMemory(opt.capacity)
        # 实例化一个策略网络对象 用于策略选择
        self.model_policy = DNN(opt, sce, scenario)
        # 实例化一个目标网络对象 用于计算目标值
        self.model_target = DNN(opt, sce, scenario)
        # 使用策略网络的参数来初始化目标网络
        self.model_target.load_state_dict(self.model_policy.state_dict())
        # 将目标网络设置为评估模式（eval()），这意味着该网络将不会主动更新其权重
        self.model_target.eval()
        # 实例化一个优化器对象，使用RMSprop算法，学习率和动量由传入的opt参数决定。这个优化器将用于更新策略网络的权重
        self.optimizer = optim.RMSprop(params=self.model_policy.parameters(), lr=opt.learningrate, momentum=opt.momentum)
    # 目的: 根据给定的场景设置代理的位置。
    # 输入: scenario: 一个场景对象，其中包含了基站（BS）的位置信息。
    # 输出: Loc_agent: 一个包含两个元素的numpy数组，表示代理的二维位置。
    def Set_Location(self, scenario):  # Initialize the location of the agent
        Loc_MBS, _ , _ = scenario.BS_Location() #从场景中获取所有基站的坐标
        Loc_agent = np.zeros(2) #初始化一个长度为2的零向量，用于存储代理的位置
        LocM = choice(Loc_MBS) #从基站位置列表中随机选择一个作为代理的参考位置
        r = self.sce.rMBS*random() # 随机生成一个与基站半径有关的距离
        theta = uniform(-pi,pi) # 生成一个在[-π, π]之间的均匀随机角度
        # 使用三角函数计算代理的新位置，并返回
        Loc_agent[0] = LocM[0] + r*np.cos(theta)
        Loc_agent[1] = LocM[1] + r*np.sin(theta) 
        return Loc_agent
    # 获取代理的当前位置
    def Get_Location(self):
        return self.location
     
    def Select_Action(self, state, scenario, eps_threshold):   # Select action for a user based on the network state
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        sample = random()       
        if sample < eps_threshold:  # epsilon-greeedy policy
            # 这是一个上下文管理器，用于指示接下来的操作不需要计算梯度。在强化学习中，我们通常不需要对所有操作都计算梯度，
            # 特别是当这些操作不涉及学习时。使用torch.no_grad()可以节省计算资源
            with torch.no_grad():
                # 这行代码使用一个预定义的模型（可能是一个深度神经网络）来评估给定状态state的Q值
                Q_value = self.model_policy(state)   # Get the Q_value from DNN
                # max(0)函数沿着第一个维度找到最大值，然后[1]获取这些最大值对应的索引，
                # 最后view(1,1)确保索引被表示为一个2D tensor
                action = Q_value.max(0)[1].view(1,1)
        else:
            # 这行代码生成一个随机的动作索引。randrange(L*K)生成一个在0到L*K-1之间的随机整数，
            # 然后这个整数被用来索引一个动作
            action = torch.tensor([[randrange(L*K)]], dtype=torch.long)
        #   根据上述条件（探索或利用），智能体返回一个随机或基于策略的动作索引
        return action      
    # 获取state-action对的回报
    def Get_Reward(self, action, action_i, state, scenario):
        BS = scenario.Get_BaseStations()
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels 

        BS_selected = action_i // K # 根据选定的动作索引转换到选定的BS
        Ch_selected = action_i % K  # 根据选定的动作索引转换到选定的信道

        Loc_diff = BS[BS_selected].Get_Location() - self.location
        # 计算BS与用户设备的距离
        distance = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))
        # 计算BS的接收功率
        Rx_power = BS[BS_selected].Receive_Power(distance)
        if Rx_power == 0.0:
            # 超出所选BS的范围，因此获得负奖励
            reward = self.sce.negative_cost
            QoS = 0  # 服务质量为０
        else: # 如果在覆盖范围内，那么我们将计算奖励值
            Interference = 0.0
            for i in range(self.opt.nagents):   # Obtain interference on the same channel
                BS_select_i = action[i] // K
                Ch_select_i = action[i] % K   # The choice of other users
                if Ch_select_i == Ch_selected:  # Calculate the interference on the same channel
                    Loc_diff_i = BS[BS_select_i].Get_Location() - self.location
                    distance_i = np.sqrt((Loc_diff_i[0]**2 + Loc_diff_i[1]**2))
                    Rx_power_i = BS[BS_select_i].Receive_Power(distance_i)
                    Interference += Rx_power_i   # Sum all the interference
            Interference -= Rx_power  # Remove the received power from interference
            # self.sce.N0 可能是噪声功率谱密度（单位为 dBm/Hz
            # self.sce.BW 是系统带宽（单位为 Hz
            # 计算的结果是总的噪声功率
            Noise = 10**((self.sce.N0)/10)*self.sce.BW
            # 计算了信号干扰加噪声比 (SINR)。Rx_power 是接收到的信号功率，Interference 是干扰功率
            # 将这两个值加起来，然后除以总的噪声功率，就得到了 SINR
            # 对应于论文中公式(3)
            SINR = Rx_power/(Interference + Noise)
            # 这一行检查 SINR 是否大于或等于一个给定的服务质量阈值 self.sce.QoS_thr。这个阈值可能是一个分贝值
            if SINR >= 10**(self.sce.QoS_thr/10):
                QoS = 1
                reward = 1
            else:
                QoS = 0   
                reward = self.sce.negative_cost
            # 这些代码与计算数据速率、利润、总成本和最终的奖励有关。这些代码当前并未执行，但可能是之前或未来开发中的功能
            """Rate = self.sce.BW * np.log2(1 + SINR) / (10**6)      # Calculate the rate of UE 
            profit = self.sce.profit * Rate
            Tx_power_dBm = BS[BS_selected].Transmit_Power_dBm()   # Calculate the transmit power of the selected BS
            cost = self.sce.power_cost * Tx_power_dBm + self.sce.action_cost  # Calculate the total cost
            reward = profit - cost """
        # 奖励值转换为PyTorch tensor格式是为了使其能够在PyTorch框架中用于后续的机器学习或深度学习操作
        reward = torch.tensor([reward])
        return QoS, reward
    # 向经验池存储经验
    def Save_Transition(self, state, action, next_state, reward, scenario):
        L = scenario.BS_Number()     # The total number of BSs
        K = self.sce.nChannel        # The total number of channels
        action = torch.tensor([[action]])
        reward = torch.tensor([reward])
        # 使用 unsqueeze(0) 方法增加一个新的维度。用于将一个标量或一维向量转换为二维张量。是为了与预期的存储格式匹配
        # 状态通常表示为一个高维的向量，其中包含了环境中与智能体决策相关的各种信息
        # 标量或一维向量，直接保存可能会导致后续处理（如神经网络输入）的困难
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        self.memory.Push(state, action, next_state, reward)
    
    def Target_Update(self):  # Update the parameters of the target network
        self.model_target.load_state_dict(self.model_policy.state_dict())
    # 优化Agent中的模型
    def Optimize_Model(self):
        # 如果记忆库中的数据量小于批量大小，则直接返回，不进行任何操作。这是为了确保有足够的数据进行训练。
        if len(self.memory) < self.opt.batch_size:
            return
        # 从记忆库中随机采样一批次的数据，并将其组合成一个批次的数据结构
        transitions = self.memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))
        # 创建了一个布尔张量 map函数和lambda表达式一起检查batch.next_state中的每个元素是否为None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
        # non_final_next_states是一个张量，包含了所有非最终状态的下一个状态
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        # 将不同属性的数据（如状态、动作、奖励）从批次的各个元素中提取出来，并合并成一个大的张量
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # 预测下一个状态的值，并与预期的状态-动作值进行比较，计算损失
        # .gather()根据之前步骤中计算出的动作索引，从目标网络的输出中收集对应的Q值
        state_action_values = self.model_policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.opt.batch_size)
        # 使用Q网络来评估non_final_next_states中的所有可能动作，
        # 然后获取每个状态-动作对的最大Q值对应的动作索引 增加一个额外的维度是为了广播操作
        next_action_batch = torch.unsqueeze(self.model_policy(non_final_next_states).max(1)[1], 1)
        # 首先使用目标Q网络来评估non_final_next_states
        next_state_values = self.model_target(non_final_next_states).gather(1, next_action_batch)
        # self.opt.gamma: 是折扣因子，通常接近于1，用于缩放未来的奖励
        # reward_batch.unsqueeze(1)增加一个新的维度，以便广播操作
        # expected_state_action_values计算预期的Q值，它是通过将未来Q值与当前奖励相加来得到的
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch.unsqueeze(1) 
        # Compute Huber loss
        # 计算平滑L1损失
        # SmoothL1损失既解决了L1损失在y-y_=0处不可导的问题，又避免了L2在误差大于1时会放大误差的问题
        # 在处理噪声或异常值时是有益的
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # Double DQN
        # 这段代码是被注释掉的代码，用于处理DQN中的非终止状态
        # 单DQN中对于非终止状态，我们使用模型直接预测下一个状态的最大值作为预期的状态值
        # 在Double DQN中，我们使用一个单独的模型进行选择操作，另一个模型进行估计操作
        """
        next_state_values[non_final_mask] = self.model_target(non_final_next_states).max(1)[0].detach()  # DQN
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        """
        # Optimize the model
        # 使用优化器对模型进行优化。首先，将梯度归零，然后反向传播损失，最后更新模型的参数。
        # 此外，为了避免梯度爆炸，对梯度进行了裁剪操作
        # 在反向传播之前，需要将之前的梯度清零。这是因为在PyTorch中，梯度是累积的，不清零会导致梯度累加
        self.optimizer.zero_grad()
        # 这是PyTorch中进行反向传播的命令。它根据损失函数计算出每个参数的梯度
        loss.backward()
        # 对每个参数进行梯度裁剪，防止梯度爆炸
        for param in self.model_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        # 使用优化器来更新模型的参数。这通常涉及到根据计算出的梯度来减小损失
        # 优化器会使用之前计算的梯度来更新模型中的权重
        # 不仅使用梯度，还使用其他技巧，如动量（Momentum）或Adam等，来加速训练过程并提高稳定性
        # 优化器可能会使用缓存或其他状态来存储与梯度计算相关的信息。
        # self.optimizer.step()方法可能会清除这些缓存或状态，为下一次迭代做准备
        # 正则化步骤，例如权重衰减（L2正则化），以防止模型过拟合。这可以在self.optimizer.step()中或之前进行
        self.optimizer.step()

            
            

            
            
            
            
            
            
            
            
            
            
            
            


        
        
        
