# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:49:45 2020

@author: liangyu

Running simuluation
"""

import copy, json, argparse
import torch
from scenario import Scenario
from agent import Agent
from dotdic import DotDic

# 如果CUDA环境可用，则将设备设置为GPU。
# 否则，将设备设置为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# opt是一个包含多个配置选项的对象或字典,它包含了运行代理所需的配置参数，例如代理的数量、学习率等
# sce是一个场景对象，包含了代理需要交互的环境信息
# scenario描述代理应该如何行动的
# device指定了计算设备，是否在CPU或GPU上运行代码
# 返回了一个Agent对象组成的数组
def create_agents(opt, sce, scenario, device):
	agents = []   # Vector of agents
	for i in range(opt.nagents):
		agents.append(Agent(opt, sce, scenario, index=i, device=device)) # Initialization, create a CNet for each agent
	return agents
# 用于执行多代理系统的episode 训练一组代理来协作完成任务
#代理们与场景进行交互 并根据交互结果进行学习并做出决策
def run_episodes(opt, sce, agents, scenario):
    # 用于跟踪全局步骤数
    global_step = 0
    # 用于跟踪当前episode的编号
    nepisode = 0
    # 一个张量，用于存储每个代理的行动
    action = torch.zeros(opt.nagents,dtype=int)
    # 一个张量，用于存储每个代理从环境中获得的奖励
    reward = torch.zeros(opt.nagents)
    # 一个张量，用于存储每个代理的QoS（质量服务）值
    QoS = torch.zeros(opt.nagents)
    # 一个张量，用于存储每个代理的目标状态
    state_target = torch.ones(opt.nagents)  # The QoS requirement
    f= open("DDQN.csv","w+")
    f.write("This includes the running steps:\n")
    while nepisode < opt.nepisodes:
        # 通过torch.zeros函数初始化state和next_state，这是表示智能体当前状态和下一个状态的张量
        state = torch.zeros(opt.nagents)  # Reset the state   
        next_state = torch.zeros(opt.nagents)  # Reset the next_state
        nstep = 0
        # 控制每个episode内的训练步数，确保在每个episode中执行不超过指定步数 opt.nsteps 的训练
        while nstep < opt.nsteps:
            # 使用线性增长的方式计算epsilon，这是一种控制智能体在训练中进行探索的策略 决定每次的动作选择
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max  # Linear increasing epsilon
                # eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-1. * nstep * (nepisode + 1)/opt.eps_decay) 
                # Exponential decay epsilon
            for i in range(opt.nagents):
                # 对于每个代理，根据当前状态、场景和epsilon值选择最优动作或随机动作
                action[i] = agents[i].Select_Action(state, scenario, eps_threshold)  # Select action
            for i in range(opt.nagents):
                # 根据所选动作和当前状态，代理从环境中获得奖励和下一个状态。这些信息被用于更新代理的Q值和进行训练
                QoS[i], reward[i] = agents[i].Get_Reward(action, action[i], state, scenario)  # Obtain reward and next state
                next_state[i] = QoS[i]
            for i in range(opt.nagents):
                # 代理保存了当前状态、动作、下一个状态和奖励的信息，并使用这些信息来优化其网络模型
                agents[i].Save_Transition(state, action[i], next_state, reward[i], scenario)  # Save the state transition
                agents[i].Optimize_Model()  # Train the model
                # 按指定周期更新目标网络的参数
                if nstep % opt.nupdate == 0:
                    agents[i].Target_Update()
            # 更新状态 在每个时间步结束后，当前状态被更新为下一个状态。这为下一个时间步做准备
            state = copy.deepcopy(next_state)
            # 如果所有代理的状态都达到了目标状态state_target，则终止当前episode的训练
            if torch.all(state.eq(state_target)):  # If QoS is satisified, break
                break
            nstep += 1
        # 在每个episode结束时，打印出当前的episode编号和训练步数。
        # 此外，这些信息也被写入到名为"DDQN.csv"的CSV文件中。这有助于后续的分析和性能评估
        print('Episode Number:', nepisode, 'Training Step:', nstep)       
     #   print('Final State:', state)
        f.write("%i \n" % nstep)
        nepisode += 1
    f.close()
                
def run_trial(opt, sce):
    # 实例化一个环境
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)  # Initialization 
    run_episodes(opt, sce, agents, scenario)    
        
if __name__ == '__main__':
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_path1', type=str,default="Config/config_1.json",
                        help='path to existing scenarios file')
    parser.add_argument('-c2', '--config_path2', type=str,default="Config/config_2.json",
                        help='path to existing options file')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    # 解析命令行参数
    args = parser.parse_args()
    # 分别读取两个配置文件并转换为字典
    # 场景
    sce = DotDic(json.loads(open(args.config_path1, 'r').read()))
    # 描述计算系统的参数
    opt = DotDic(json.loads(open(args.config_path2, 'r').read()))
    # 循环运行试验：根据用户指定的次数（默认为1次）运行试验
    for i in range(args.ntrials):
        trial_result_path = None
        # 每次试验中都使用相同的配置，而不是修改原始的配置
        # copy.deepcopy 用于创建对象（这里是字典）的深拷贝。
        # 这意味着每次试验都会使用新的、独立的配置，不会影响到原始的配置
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt, trial_sce)
















