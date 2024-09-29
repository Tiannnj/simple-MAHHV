import time

import numpy as np

from mvgym.envs.VORenv.vorenv import vorenv
class MahhvEnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        team_size = self.agent_num
        grid_size = (15, 15)
        self.env = vorenv(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
        self.obs_dim = 25  # 设置智能体的观测维度 # set the observation dimension of agents
        self.ou_obs_dim = 43
        self.ru_obs_dim = 25
        self.action_dim = self.env.action_space[0].n  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
        self.ou_action_dim = self.env.ou_action_space[0].n
        self.ru_action_dim = self.env.ru_action_space[0].n

    def reset(self, x):
        s = self.env.Mahhv_reset(x)
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim) observation data
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.array(s[i]) #np.random.random(size=(14,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions, x):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        # self.env.render("human")
        time.sleep(0.4)
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
#        print('actions', actions)  #  [[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
        # action_index = [int(np.where(act==1)[0][0]) for act in actions]
        next_s, r, done = self.env.Mahhv_step(actions,  x)
#        print('next_s, r', next_s, r)
        for i in range(self.agent_num):
            # r[agent_i] + 100 if info['win'] else r[agent_i] - 0.1
            sub_agent_obs.append(np.array(next_s[i]))
            sub_agent_reward.append(np.array(r[i]))
            sub_agent_done.append(done[i])

        return [sub_agent_obs, sub_agent_reward, sub_agent_done]
