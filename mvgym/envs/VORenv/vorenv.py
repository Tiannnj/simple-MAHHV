# -*- coding: utf-8 -*-

import copy
import logging
import random
import math
import collections

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, write_cell_text

logger = logging.getLogger(__name__)


class vorenv(gym.Env):
    """
    We simulate IoV environment
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    # 2 OU agents and 4RU agents
    def __init__(self, grid_shape=(15, 15), n_agents=6, n_opponents=5, init_health=3, full_observable=False,
                 step_cost=0, max_steps=100, n_v = 10, n_o_agents = 2, n_r_agents = 4):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_opponents = n_opponents
        self._max_steps = max_steps
        self._step_cost = step_cost
        self._step_count = None
        self.n_v = n_v
        self.n_o_agents = n_o_agents
        self.n_r_agents = n_r_agents


        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(8) for _ in range(self.n_o_agents)] + [spaces.Discrete(8) for _ in range(self.n_r_agents)])

        # init vehicles in each time step, set the vehicles run in a 500m road, the speed of each vehicle is set as 10m/s, conencted with D, C, L
        self.n_v = 10
        self.v_info = {_: [ _ * 50 + 10, 0, 0, random.uniform(0.1,0.6), random.uniform(0.1,0.6), random.uniform(20,100)] for _ in range(self.n_v)}
#        print('self.v_info',self.v_info)

        # init OU in each time step, OU can observe itself's location, the vehicles info in it range, the task number it has received, number of tasks assigned to RU, OU service fairness
        self.n_o_agents = 2
        self.o_location = np.array([[150, 0, 50],[350, 0, 50]])
        self.o_receive = np.array([[0], [0]])
        self.o_tra = np.array([[0, 0],[0, 0]])
        self.o_fairness = np.array([[0], [0]])
        self.o_v = np.concatenate((np.array(list(self.v_info.values()))[0:6], np.array(list(self.v_info.values()))[4:10]))
        self.o_info = {_: np.concatenate((self.o_location[_], self.o_v[_], self.o_receive[_],self.o_tra[_])) for _ in range(self.n_o_agents)}
        self.ou_action_space = MultiAgentActionSpace([spaces.Discrete(8) for _ in range(self.n_o_agents)])

        # init RU in each time step, RU can observe itself's location, the vehicles info in it range, number of tasks assigned to RSU associated to it, RSU service fairness
        self.n_r_agents = 4
        self.r_location = np.array([[100, 0, 50], [200, 0, 50], [300, 0, 50], [400, 0, 50]])

        self.r_v = np.concatenate(([np.array(list(self.v_info.values()))[0:6, 3:6].reshape(-1)], [np.array(list(self.v_info.values()))[0:6, 3:6].reshape(-1)],
                                   [np.array(list(self.v_info.values()))[4:10, 3:6].reshape(-1)], [np.array(list(self.v_info.values()))[4:10, 3:6].reshape(-1)]))
        self.r_assign = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.r_info = {_: np.concatenate((self.r_location[_], self.r_v[_],  self.r_assign[_])) for _ in range(self.n_r_agents)}
        self.ru_action_space = MultiAgentActionSpace([spaces.Discrete(6 + 2) for _ in range(self.n_r_agents)])


        # init RSU in each time step, RSU can observe itself's location, RSU’s CPU frequency
        self.n_MeNB = 8
        self.m_location = np.array([[100, -50, 50], [100, 50, 50], [200, -50, 50], [200, 50, 50],
                           [300, -50, 50], [300, 50, 50], [400, -50, 50], [400, 50, 50]])
        self.m_cpu = np.array([[250], [300], [250], [300],
                      [250], [300], [250], [300]])


        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_prev_pos = {_: None for _ in range(self.n_agents)}
        self.opp_pos = {_: None for _ in range(self.n_agents)}
        self.opp_prev_pos = {_: None for _ in range(self.n_agents)}

        self._init_health = init_health
        self.agent_health = {_: None for _ in range(self.n_agents)}
        self.opp_health = {_: None for _ in range(self._n_opponents)}
        self._agent_dones = [None for _ in range(self.n_agents)]
        self._agent_cool = {_: None for _ in range(self.n_agents)}
        self._opp_cool = {_: None for _ in range(self._n_opponents)}
        self._total_episode_reward = None
        self.viewer = None
        self.full_observable = full_observable

        # # 5 * 5 * (type, id, health, cool, x, y)
        # self._obs_low = np.repeat([-1., 0., 0., -1., 0., 0.], 5 * 5)
        # self._obs_high = np.repeat([1., n_opponents, init_health, 1., 1., 1.], 5 * 5) 
        # observation space for OU
        self._obs_low = np.repeat([-10000], 43)
        self._obs_high = np.repeat([10000], 43)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])



    def get_action_meanings(self, agent_i=None):
        action_meaning = []
        for _ in range(self.n_agents):
            meaning = [ACTION_MEANING[i] for i in range(5)]
            meaning += ['Attack Opponent {}'.format(o) for o in range(self._n_opponents)]
            action_meaning.append(meaning)
        if agent_i is not None:
            assert isinstance(agent_i, int)
            assert agent_i <= self.n_agents

            return action_meaning[agent_i]
        else:
            return action_meaning

    @staticmethod
    def _one_hot_encoding(i, n):
        x = np.zeros(n)
        x[i] = 1
        return x.tolist()

    def get_agent_obs(self):
        """
        When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
        encoding its unique ID, team ID, location, health points and cooldown.
        A model controlling an agent also sees other agents in its visual range (5 × 5 surrounding area).
        :return:
        """
        _obs = []
        total_obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # _agent_i_obs = self._one_hot_encoding(agent_i, self.n_agents)
            # _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            # _agent_i_obs += [self.agent_health[agent_i]]
            # _agent_i_obs += [1 if self._agent_cool else 0]  # flag if agent is cooling down

            # team id , unique id, location,health, cooldown

            _agent_i_obs = np.zeros((6, 5, 5))
            for row in range(0, 5):
                for col in range(0, 5):

                    if self.is_valid([row + (pos[0] - 2), col + (pos[1] - 2)]) and (
                            PRE_IDS['empty'] not in self._full_obs[row + (pos[0] - 2)][col + (pos[1] - 2)]):
                        x = self._full_obs[row + pos[0] - 2][col + pos[1] - 2]
                        _type = 1 if PRE_IDS['agent'] in x else -1
                        _id = int(x[1:]) - 1  # id
                        _agent_i_obs[0][row][col] = _type
                        _agent_i_obs[1][row][col] = _id
#                         print('type', type, '_type', _type)
                        _agent_i_obs[2][row][col] = self.agent_health[_id] if _type == 1 else self.opp_health[_id]
                        _agent_i_obs[3][row][col] = self._agent_cool[_id] if _type == 1 else self._opp_cool[_id]
                        _agent_i_obs[3][row][col] = 1 if _agent_i_obs[3][row][col] else -1  # cool/uncool

                        _agent_i_obs[4][row][col] = pos[0] / self._grid_shape[0]  # x-coordinate
                        _agent_i_obs[5][row][col] = pos[1] / self._grid_shape[1]  # y-coordinate

            _agent_i_obs = _agent_i_obs.flatten().tolist()
            _obs.append(_agent_i_obs)

        # 每个OU状态生成
        for agent_o in range(self.n_o_agents):
            _agent_o_obs = np.zeros(43)
            _agent_o_obs[0: 3] = self.o_location[agent_o]   # OU location
#            print('agent_o', agent_o)
            _agent_o_obs[3: 39] = self.o_v[agent_o]     # tasks' information
            _agent_o_obs[39: 41] = self.o_receive.flatten()
            _agent_o_obs[41: 43] = self.o_tra[agent_o]   # number of tasks assigned to RU
            _agent_o_obs = _agent_o_obs.flatten().tolist()
            total_obs.append(_agent_o_obs)
#            print(' _agent_o_obs', _agent_o_obs)

        # 每个RU状态生成
        for agent_r in range(self.n_r_agents):
            _agent_r_obs = np.zeros(43)
            _agent_r_obs[0: 3] = self.m_location[agent_r]   # RU location
            _agent_r_obs[3: 21] = self.r_v[agent_r]     # tasks' information
            _agent_r_obs[21: 23] = self.r_assign[agent_r]    # number of tasks assigned to RSU associated to it
            _agent_r_obs = _agent_r_obs.flatten().tolist()
            total_obs.append(_agent_r_obs)
#            print('_agent_r_obs', _agent_r_obs)

#        print('_obs', _obs)
#        print('total_obs', total_obs)
        return _obs

    def Mahhv_get_agent_obs(self):
        """
        When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
        encoding its unique ID, team ID, location, health points and cooldown.
        A model controlling an agent also sees other agents in its visual range (5 × 5 surrounding area).
        :return:
        """
        _obs = []
        total_obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # _agent_i_obs = self._one_hot_encoding(agent_i, self.n_agents)
            # _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            # _agent_i_obs += [self.agent_health[agent_i]]
            # _agent_i_obs += [1 if self._agent_cool else 0]  # flag if agent is cooling down

            # team id , unique id, location,health, cooldown

            _agent_i_obs = np.zeros((6, 5, 5))
            for row in range(0, 5):
                for col in range(0, 5):
                    if self.is_valid([row + (pos[0] - 2), col + (pos[1] - 2)]) and (
                            PRE_IDS['empty'] not in self._full_obs[row + (pos[0] - 2)][col + (pos[1] - 2)]):
                        x = self._full_obs[row + pos[0] - 2][col + pos[1] - 2]
                        _type = 1 if PRE_IDS['agent'] in x else -1
                        _id = int(x[1:]) - 1  # id
                        _agent_i_obs[0][row][col] = _type
                        _agent_i_obs[1][row][col] = _id
                        #                         print('type', type, '_type', _type)
                        _agent_i_obs[2][row][col] = self.agent_health[_id] if _type == 1 else self.opp_health[_id]
                        _agent_i_obs[3][row][col] = self._agent_cool[_id] if _type == 1 else self._opp_cool[_id]
                        _agent_i_obs[3][row][col] = 1 if _agent_i_obs[3][row][col] else -1  # cool/uncool

                        _agent_i_obs[4][row][col] = pos[0] / self._grid_shape[0]  # x-coordinate
                        _agent_i_obs[5][row][col] = pos[1] / self._grid_shape[1]  # y-coordinate

            _agent_i_obs = _agent_i_obs.flatten().tolist()
            _obs.append(_agent_i_obs)

        # 每个OU状态生成
        for agent_o in range(self.n_o_agents):
            _agent_o_obs = np.zeros(43)
            _agent_o_obs[0: 3] = self.o_location[agent_o]  # OU location
            _agent_o_obs[3: 39] = self.o_v[agent_o]  # tasks' information
            _agent_o_obs[39: 41] = self.o_receive[:].flatten() # number of tasks received by both OU
            _agent_o_obs[41: 43] = self.o_tra[agent_o]  # number of tasks assigned to RU by each OU
            _agent_o_obs = _agent_o_obs.flatten().tolist()
            total_obs.append(_agent_o_obs)

        # 每个RU状态生成
        for agent_r in range(self.n_r_agents):
            _agent_r_obs = np.zeros(43)
            _agent_r_obs[0: 3] = self.m_location[agent_r]  # RU location
            _agent_r_obs[3: 21] = self.r_v[agent_r]  # tasks' information
            _agent_r_obs[21: 23] = self.r_assign[agent_r]  # number of tasks assigned to RSU associated to it
            _agent_r_obs = _agent_r_obs.flatten().tolist()
            total_obs.append(_agent_r_obs)
        return total_obs


    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_opp_view(self, opp_i):
        self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __init_full_obs(self):
        """ Each team consists of m = 10 agents and their initial positions are sampled uniformly in a 5 × 5
        square.
        """
        self._full_obs = self.__create_grid()

        # select agent team center
        # Note : Leaving space from edges so as to have a 5x5 grid around it
        agent_team_corner = random.randint(0, int(self._grid_shape[0] / 2)), random.randint(0, int(self._grid_shape[1] / 2))
        agent_pos_index = random.sample(range(25), self.n_agents)
        # randomly select agent pos
        for agent_i in range(self.n_agents):
            pos = [int(agent_pos_index[agent_i] / 5) + agent_team_corner[0], agent_pos_index[agent_i] % 5 + agent_team_corner[1]]
#             print(pos)
            while True:
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.agent_prev_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    self.__update_agent_view(agent_i)
                    break
                pos = [random.randint(agent_team_corner[0], agent_team_corner[0] + 4),
                       random.randint(agent_team_corner[1], agent_team_corner[1] + 4)]

        # select opponent team center
        while True:
            pos = random.randint(agent_team_corner[0], self._grid_shape[0] - 5), random.randint(agent_team_corner[1], self._grid_shape[1] - 5)
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                opp_team_corner = pos
                break
                
        opp_pos_index = random.sample(range(25), self._n_opponents)

        # randomly select opponent pos
        for opp_i in range(self._n_opponents):
            pos = [int(opp_pos_index[agent_i] / 5) + opp_team_corner[0], opp_pos_index[agent_i] % 5 + opp_team_corner[1]]
            while True:
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.opp_prev_pos[opp_i] = pos
                    self.opp_pos[opp_i] = pos
                    self.__update_opp_view(opp_i)
                    break
                pos = [random.randint(opp_team_corner[0], opp_team_corner[0] + 4),
                       random.randint(opp_team_corner[1], opp_team_corner[1] + 4)]

        self.__draw_base_img()

        # initial the Env's information
        total_obs = []
        # 每个OU状态生成
        for agent_o in range(self.n_o_agents):
            _agent_o_obs = np.zeros(43)
            _agent_o_obs[0: 3] = self.o_location[agent_o]   # OU location
#            print('agent_o', agent_o)
            _agent_o_obs[3: 39] = self.o_v[agent_o]     # tasks' information
            _agent_o_obs[39: 41] = self.o_tra[agent_o]   # number of tasks assigned to RU
            _agent_o_obs[41] = 0    # OU service fairness
            _agent_o_obs[42] = 0    # RU offloaded fairness
            _agent_o_obs = _agent_o_obs.flatten().tolist()
            total_obs.append(_agent_o_obs)
#            print(' _agent_o_obs', _agent_o_obs)

        # 每个RU状态生成
        for agent_r in range(self.n_r_agents):
            _agent_r_obs = np.zeros(24)
            _agent_r_obs[0: 3] = self.m_location[agent_r]   # RU location
            _agent_r_obs[3: 21] = self.r_v[agent_r]     # tasks' information
            _agent_r_obs[21:23] = self.r_assign[agent_r]    # number of tasks assigned to RSU associated to it
            _agent_r_obs[23: 34] = self.r_fairness[agent_r]     # RSU service fairness
            _agent_r_obs = _agent_r_obs.flatten().tolist()
            total_obs.append(_agent_r_obs)
#            print('_agent_r_obs', _agent_r_obs)

        inital_bos = total_obs
        return inital_bos


    def reset(self):
        # UAV initial state, including 
        # Vehicles' locations, task information;
        # OUs' locations, total bandwidth for receiving and offloading tasks(evenly allocation), OU service fairness
        # RUs' locations, total bandwidht for offloading tasks(finish bandwidth allocation)
        
        # initial count
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]

        # initial vehicles' states
        
        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self._agent_cool = {_: True for _ in range(self.n_agents)}
        self._opp_cool = {_: True for _ in range(self._n_opponents)}
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.__init_full_obs()
        return self.get_agent_obs()

    def Mahhv_reset(self):
        # UAV initial state, including
        # Vehicles' locations, task information;
        # OUs' locations, total bandwidth for receiving and offloading tasks(evenly allocation), OU service fairness
        # RUs' locations, total bandwidht for offloading tasks(finish bandwidth allocation)

        # initial count
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]

        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self._agent_cool = {_: True for _ in range(self.n_agents)}
        self._opp_cool = {_: True for _ in range(self._n_opponents)}
        self._agent_dones = [False for _ in range(self.n_agents)]

        # initial environment
        # init vehicles in each time step, set the vehicles run in a 500m road, the speed of each vehicle is set as 10m/s
        self.n_v = 10
        self.v_info = {
            _: [_ * 50 + 10, 0, 0, random.uniform(0.1, 0.6), random.uniform(0.1, 0.6), random.uniform(20, 100)] for _ in
            range(self.n_v)}
#        print('self.v_info', self.v_info)

        # init OU in each time step, OU can observe itself's location, the vehicles info in it range, the task number it has received, number of tasks assigned to RU, OU service fairness
        self.n_o_agents = 2
        self.o_location = np.array([[150, 0, 50], [350, 0, 50]])
        self.o_receive = np.array([[0], [0]])
        self.o_tra = np.array([[0, 0], [0, 0]])
        self.o_fairness = np.array([[0], [0]])
        self.o_v = np.concatenate(([np.array(list(self.v_info.values()))[0:6].reshape(-1)],
                                   [np.array(list(self.v_info.values()))[4:10].reshape(-1)]))
        self.o_info = {_: np.concatenate(
            (self.o_location[_], self.o_v[_], self.o_receive.flatten(), self.o_tra[_], self.o_fairness[_])) for _ in
                       range(self.n_o_agents)}
#        print('self.o_info', self.o_info)

        # init RU in each time step, RU can observe itself's location, the vehicles info in it range, number of tasks assigned to RSU associated to it, RSU service fairness
        self.n_r_agents = 4
        self.r_location = [[100, 0, 50], [200, 0, 50], [300, 0, 50], [400, 0, 50]]

        self.r_v = np.concatenate(([np.array(list(self.v_info.values()))[0:6, 3:6].reshape(-1)],
                                   [np.array(list(self.v_info.values()))[0:6, 3:6].reshape(-1)],
                                   [np.array(list(self.v_info.values()))[4:10, 3:6].reshape(-1)],
                                   [np.array(list(self.v_info.values()))[4:10, 3:6].reshape(-1)]))
        self.r_assign = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.r_fairness = [[0], [0], [0], [0]]
        self.r_info = {_: np.concatenate((self.m_location[_], self.r_v[_], self.r_assign[_], self.r_fairness[_])) for _
                       in range(self.n_r_agents)}
#        print('self.r_info', self.r_info)

        # init MeNB in each time step, RSU can observe itself's location, RSU’s CPU frequency
        self.n_MeNB = 8
        self.m_location = [[100, -50, 50], [100, 50, 50], [200, -50, 50], [200, 50, 50],
                           [300, -50, 50], [300, 50, 50], [400, -50, 50], [400, 50, 50]]
        self.m_cpu = [[250], [300], [250], [300],
                      [250], [300], [250], [300]]

        # initial agents' observation
        self.__init_full_obs()
        return self.Mahhv_get_agent_obs()


    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_opp_pos(self, opp_i, move):

        curr_pos = copy.copy(self.opp_pos[opp_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.opp_pos[opp_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_opp_view(opp_i)

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    @staticmethod
    def is_visible(source_pos, target_pos):
        """
        Checks if the target_pos is in the visible range(5x5)  of the source pos

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) \
               and (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)

    @staticmethod
    def is_fireable(source_pos, target_pos):
        """
        Checks if the target_pos is in the firing range(3x3)

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 1) <= target_pos[0] <= (source_pos[0] + 1) \
               and (source_pos[1] - 1) <= target_pos[1] <= (source_pos[1] + 1)

    def reduce_distance_move(self, source_pos, target_pos):

        # Todo: makes moves Enum
        _moves = []
        if source_pos[0] > target_pos[0]:
            _moves.append('UP')
        elif source_pos[0] < target_pos[0]:
            _moves.append('DOWN')

        if source_pos[1] > target_pos[1]:
            _moves.append('LEFT')
        elif source_pos[1] < target_pos[1]:
            _moves.append('RIGHT')

        move = random.choice(_moves)
        for k, v in ACTION_MEANING.items():
            if move.lower() == v.lower():
                move = k
                break
        return move

    @property
    def opps_action(self):
        """
        Opponent bots follow a hardcoded policy.

        The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
        it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
        is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

        :return:
        """

        visible_agents = set([])
        opp_agent_distance = {_: [] for _ in range(self._n_opponents)}

        for opp_i, opp_pos in self.opp_pos.items():
            for agent_i, agent_pos in self.agent_pos.items():
                if agent_i not in visible_agents and self.agent_health[agent_i] > 0 \
                        and self.is_visible(opp_pos, agent_pos):
                    visible_agents.add(agent_i)
                distance = abs(agent_pos[0] - opp_pos[0]) + abs(agent_pos[1] - opp_pos[1])  # manhattan distance
                opp_agent_distance[opp_i].append([distance, agent_i])

        opp_action_n = []
        for opp_i in range(self._n_opponents):
            action = None
            for _, agent_i in sorted(opp_agent_distance[opp_i]):
                if agent_i in visible_agents:
                    if self.is_fireable(self.opp_pos[opp_i], self.agent_pos[agent_i]):
                        action = agent_i + 5
                    else:
                        action = self.reduce_distance_move(self.opp_pos[opp_i], self.agent_pos[agent_i])
                    break
            if action is None:
                logger.info('No visible agent for enemy:{}'.format(opp_i))
                action = random.choice(range(5))
            opp_action_n.append(action)


        return opp_action_n


    def Mahhv_step(self, agents_action):
        rewards = [self._step_cost for _ in range(self.n_agents)]
        # remain ddl on each OU
        real_state = []
        ou_remain_ddl = [[0] * 6, [0] * 6]
        for o_num in range(0, self.n_o_agents):
            for v_num in range(0, 6):
                ou_remain_ddl[o_num][v_num] = self.o_v[o_num][6 * v_num + 5]
#        print('ou_remain_ddl', ou_remain_ddl)
        fixed = [[1], [1], [1], [1]]
        agents_action[0] = np.r_[fixed, agents_action[0]]
        agents_action[1] = np.r_[agents_action[1][0:2], fixed, agents_action[1][-6:]]
        for share_v in range(0, 2):
            if (agents_action[0][4 + share_v] + agents_action[1][share_v]) != 1 and agents_action[0][4 + share_v] == 0:
                agents_action[1][share_v] = 1
            if (agents_action[0][4 + share_v] + agents_action[1][share_v]) != 1 and agents_action[0][4 + share_v] == 1:
                agents_action[1][share_v] = 0
        ou_action = agents_action[0:2]
        ru_action = agents_action[-4:]
        ou_delay = [[0] * 6, [0] * 6]
        self.o_v_new = self.o_v
        self.r_m_new = self.o_v
        rewards_ou = [0 for _ in range(self.n_o_agents)]
        rewards_ru = [0 for _ in range(self.n_r_agents)]
        r_receive = np.zeros((4, 6))
        # take actions for OU agents
        for o_agent_num, o_action in enumerate(ou_action):
            band_o2v = 5
            band_o2r = 1
            # get the observation for OU
            o_pre_state = self.Mahhv_get_agent_obs()[o_agent_num]
            real_state.append(o_pre_state)
            print('o_pre_state', o_pre_state)
            num_o_receive = self.Mahhv_get_agent_obs()[o_agent_num][-4:-2]
            num_o_tra = o_pre_state[-2:]
            num_r_rtr = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

            # communication channel setting
            for v in range (0, 6):
                # distance between vehicle v and OU o
                dis_o2v = np.sqrt((np.array(list(self.v_info.values()))[v + o_agent_num * 4, 0:3][0] - self.o_location[o_agent_num][0]) ** 2 +
                         (np.array(list(self.v_info.values()))[v + o_agent_num * 4, 0:3][1] - self.o_location[o_agent_num][1]) ** 2 +
                         (np.array(list(self.v_info.values()))[v + o_agent_num * 4 , 0:3][2] - self.o_location[o_agent_num][2]) ** 2)
                # height of OU
                height_o = 50
                # the angle used to calculate the los probability (r2m)
                ars = dis_o2v / height_o
                # path loss
                PL = (20 * math.log10(83.78 * dis_o2v + 0.00001)) + 1  # 计算path loss
                # sum of vehicles to serve
                num_v = sum(o_action[0:6])
                # average bandwidth
                band_even_v = band_o2v / num_v
                if num_v == 0 :
                    band_even_v = 0
                # signal noise rate
                SNR = (0.2 * 1000 * 10 ** (- PL / 10)) / ((10 ** -17.4) * (band_even_v * 10 ** 6 + 0.000001))
                rate_v2o = band_even_v * math.log2(1 + SNR)
                # delay from v2o
                delay_v2o = (self.o_v[o_agent_num][3 + v * 6]) / rate_v2o
                # transmit task to RUs
                for r in range(0, 2):
                    # record o_r_v condition for each RU
                    # judge how many tasks RU has to retransmit
                    if o_agent_num == 0:
                        r = r
                    else:
                        r = r + 2
                    # task amount that each RU received
                    if r == 0 and o_action[0:6][v] == 1 and o_action[-6:][v] == 0:
                        r_receive[r][v] = 1
                    if r == 1 and o_action[0:6][v] == 1 and o_action[-6:][v] == 1:
                        r_receive[r][v] = 1
                    if r == 2 and o_action[0:6][v] == 1 and o_action[-6:][v] == 0:
                        r_receive[r][v] = 1
                    if r == 3 and o_action[0:6][v] == 1 and o_action[-6:][v] == 1:
                        r_receive[r][v] = 1
                    # print('r_receive', r_receive)
                    dis_o2r = np.sqrt((self.r_location[r][0] - self.o_location[o_agent_num][0]) ** 2 +
                        (self.r_location[r][1] - self.o_location[o_agent_num][1]) ** 2 +
                        (self.r_location[r][2] - self.o_location[o_agent_num][2]) ** 2)
                    # path loss
                    PL = (20 * math.log10(83.78 * dis_o2r + 0.00001)) + 1
                    # sum of task allocated by OU o_agent_num which is equal to the sum of vehicles to serve
                    num_r = sum(o_action[0:6])
                    # average bandwidth
                    if o_action[0:6][v] == 1:
                        band_even_r = band_o2r / num_r
                    else:
                        band_even_r = 0
                    # signal noise rate
                    SNR = (1 * 1000 * 10 ** (- PL / 10)) / ((10 ** -17.4) * (band_even_r * 10 ** 6 + 0.000001))
                    rate_o2r = band_even_r * math.log2(1 + SNR)  # Mb/s
#                    print('rate_o2r', rate_o2r)
                    # delay from o to r
                    if rate_o2r > 0:
                        delay_o2r = (self.o_v[o_agent_num][3 + v * 6] * 8) / rate_o2r
                    else:
                        delay_o2r = 0
                    # the total delay in the stage 1
#                    print('delay_v2o', delay_v2o, delay_o2r)
                    # in case that OU choose to receive v task
                    if o_action[v] == 1:
                        ou_total_delay = delay_v2o + delay_o2r
                        ou_delay[o_agent_num][v] = ou_total_delay
                    # in case that RU is choosen to offloaded
                    if rate_o2r > 0:
                        ou_remain_ddl[o_agent_num][v] = ou_remain_ddl[o_agent_num][v] - ou_total_delay
                        # Generate new vehicle task information
                        self.o_v_new[o_agent_num][v * 6 + 5] = self.o_v_new[o_agent_num][v * 6 + 5] - ou_total_delay


            o_receive_tmp = sum(o_action[0:6])
            # OU - number for receiving increase
            num_o_receive[o_agent_num] = int(num_o_receive[o_agent_num] + o_receive_tmp)
            # tmp array to record the RU receive task amount in one step for one OU
            o_tra_tmp = [0, 0]
            for r in range (0, 2):
                o_tra_tmp[r] = sum(r_receive[o_agent_num * 2 + r])
                #OU - number for assign increase
                num_o_tra[r] = num_o_tra[r] + o_tra_tmp[r]

            # The fairness in the OU observation state has to be changed
#            print('num_o_receive[o_agent_num]',num_o_receive[o_agent_num])
            self.o_receive[o_agent_num] = num_o_receive[o_agent_num]
            for r in range (0, 2):
                self.o_tra[o_agent_num][r] = num_o_tra[r]


        # take actions for RU agents  ru_action = [0,1]
        # array to save the ddl for each ru
        r_ddl = [0, 0, 0, 0]
        self.r_m_new = self.o_v_new
        for r_agent_num, r_action in enumerate(ru_action):
            if r_agent_num < 2:
                r_o = 0
            else:
                r_o = 1
            band_r2m = 2
            # get the observation for RU
            r_pre_state = self.Mahhv_get_agent_obs()[2 + r_agent_num]
            for v_tmp in range(0, 6):
                r_pre_state[3 + v_tmp * 3 + 2] = self.r_m_new[r_o][v_tmp * 6 + 5]
            # Modify the state to keep the v task not received by RU as 0 (D, C, L)
            for v in range (0, 6):
                r_pre_state[3 + v * 3 : 3 + v * 3 + 3] = np.multiply(r_pre_state[3 + v * 3 : 3 + v * 3 + 3], r_receive[r_agent_num][v])
            real_state.append(r_pre_state)
            num_r_rtr[r_agent_num] = r_pre_state[21:23]
            # communication channel setting
            for v in range (0, 6):
                # handle the task received by OU 1 assigned to RU 0 or RU 1
                if r_receive[r_agent_num][v] == 1:
                    # Target MeNB m, distance between RU r and MeNB m
                    m_index = int(r_agent_num * 2 + r_action[v])
                    dis_r2m = np.sqrt(
                        (self.r_location[r_agent_num][0] - self.m_location[m_index][0]) ** 2 +
                        (self.r_location[r_agent_num][1] - self.m_location[m_index][1]) ** 2 +
                        (self.r_location[r_agent_num][2] - self.m_location[m_index][2]) ** 2)
                    # height of OU
                    height_o = 50
                    # the angle used to calculate the los probability (r2m)
                    ars = dis_r2m / height_o
                    # LoS probability
                    plos = 1 / (1 + 9.6 * math.exp(-0.16 * (90 * np.arcsin(ars) / (math.pi / 2) - 9.6)))
                    # sum of task allocated by RU r_agent_num which is equal to the task OU assigns to RU r_agent_num
                    num_m = sum(r_receive[r_agent_num][0:6])
                    # average bandwidth
                    band_even_m = band_r2m / num_m
                    # path loss
                    PL = (20 * math.log10(83.78 * dis_r2m + 0.00001)) + 1 * plos + 20 * (1 - plos)  # 计算path loss
                    # signal noise rate
                    SNR = (0.2 * 1000 * 10 ** (- PL / 10)) / ((10 ** -17.4) * (band_even_m * 10 ** 6))
                    rate_r2m = band_even_m * math.log2(1 + SNR)
                    # record the re-tra times
                    if r_action[v] == 0:
                        num_r_rtr[r_agent_num][0] = num_r_rtr[r_agent_num][0] + 1
                    if r_action[v] == 1:
                        num_r_rtr[r_agent_num][1] = num_r_rtr[r_agent_num][1] + 1
                    # calculate the retransmit delay
                    delay_r2m = (r_pre_state[3 + v * 3] * 8)/ rate_r2m
                    # calculate the compute delay GHz/GHz
                    delay_m_compute = r_pre_state[3 + v * 3 + 1] / self.m_cpu[int(r_agent_num * 2 + r_action[v])]
                    rm_total_delay = delay_r2m + delay_m_compute
                    # Generate the final ddl information
                    # np.array(list(self.r_m_new()))[v, 5] = np.array(list(self.o_v_new.values()))[v, 5] - rm_total_delay
                    r_ddl[r_agent_num] = r_ddl[r_agent_num] + r_pre_state[3 + v * 3 + 2] - rm_total_delay

            # The fairness in the RU observation state has to be changed
            self.r_assign[r_agent_num] = num_r_rtr[r_agent_num]

        # calculate the reward for each OU
        x = 0
        y = 0
        for agent_o in range(0, self.n_o_agents):
            for i in self.o_receive:
                x = x + i ** 2
            f_o_rec = (sum(self.o_receive)) ** 2 / (2 * x + 0.0000001)
            for j in self.o_tra[agent_o]:
                y = y + j ** 2
            f_o_tra = (sum(self.o_tra[agent_o])) ** 2 / (2 * y + 0.0000001)
            o_delay = sum(ou_remain_ddl[agent_o])
            rewards_ou[agent_o] = float(o_delay * f_o_rec * f_o_tra)
        # print('rewards_ou', rewards_ou)

        # calculate the reward for each RU
        z = 0
        for agent_r in range(0, self.n_r_agents):
            for k in self.r_assign[agent_r]:
                z = z + k ** 2
            f_r_rtr = (sum(self.r_assign[agent_r])) ** 2 / (2 * z + 0.0000001)
            rewards_ru[agent_r] = float(r_ddl[agent_r] * f_r_rtr)
        # print('rewards_ru', rewards_ru)

        # Generate the total reward
        rewards = rewards_ou + rewards_ru
        # print('rewards', rewards)

        # Adjust new environmental information
        self.Mahhv_reset()
        for o in range(0, self.n_o_agents):
            self.o_receive[o] = num_o_receive[o]
        for o in range(0, self.n_o_agents):
            self.o_tra[o_agent_num][o] = num_o_tra[o]
        for r in range(0, self.n_r_agents):
            self.r_assign[r] = num_r_rtr[r]

        # new_state = np.array(self.Mahhv_reset())
        # note the test count in the previous steps
        # new_state[0 :  self.n_o_agents, -4: -2] = num_o_receive[:]
        # for xx in range (0, self.n_o_agents):
        #     new_state[0 : xx, -2: ] = self.o_tra[xx]
        # for yy in range (0, self.n_r_agents):
        #     new_state[2 : 2 + yy, -2: ] = self.r_assign[yy]
        #
        # for i in range(self.n_agents):
        #     self._total_episode_reward[i] += rewards[i]

        real_state = o_pre_state + r_pre_state
        print('real_state', real_state)
        # print('new_state', self.Mahhv_get_agent_obs())
        return self.Mahhv_get_agent_obs(), rewards, self._agent_dones

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 15

WALL_COLOR = 'black'
AGENT_COLOR = 'red'
OPPONENT_COLOR = 'blue'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'wall': 'W',
    'empty': 'E',
    'agent': 'A',
    'opponent': 'X',
}
