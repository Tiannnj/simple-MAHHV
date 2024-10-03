import time
import os
import numpy as np
from itertools import chain
import torch
import imageio
from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner
import csv
from pathlib import Path
import random


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        start = time.time()
        self.num_env_steps = 10000000
        self.episode_length = 100
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            print('********episode********', episode)
            self.warmup()
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Observe reward and next obs
                # print('******step******', step)
                obs, rewards, dones = self.envs.step(actions_env, step)


                data = (
                    obs,
                    rewards,
                    dones,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MyEnv":
                    for agent_id in range(self.num_agents):
                        # idv_rews = []
                        # for info in infos:
                        #     if "individual_reward" in info[agent_id].keys():
                        #         idv_rews.append(info[agent_id]["individual_reward"])
                        # train_infos[agent_id].update({"individual_rewards": np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length
                            }
                        )
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print('********eval*********')
                self.eval(episode, total_num_steps,)

    def warmup(self):
        # reset env
        D_array = np.random.uniform(0.1, 0.6, 100)
        C_array = np.random.uniform(0.1, 0.6, 100)
        L_array = np.random.uniform(20, 200, 100)
        self.v_info_all = {_: [_ * 50 + 10, 0, 0, random.uniform(0.1, 0.6), random.uniform(0.1, 0.6), random.uniform(20, 200)] for
                _ in range(1000)}
        obs = self.envs.reset(0)  # shape = [env_num, agent_num, obs_dim]
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]

        share_obs_group1 = np.array(list(obs[0][0][0:3])+ list(obs[0][1][0:3]) + list(np.array(obs[0][0][3:21]))+
                            list(np.array(obs[0][1][9:21]))+ list(np.array(obs[0][0][-3:-2]))+ list(np.array(obs[0][1][-3:-2])) +
                            list(np.array(obs[0][0][-2:]))+list(np.array(obs[0][1][-2:])))
        share_obs_group1 = np.array(share_obs_group1)

        share_obs_group2 = np.array(list(obs[0][2][0:3]) + list(obs[0][3][0:3]) + list(np.array(obs[0][2][3:21]))
                            + list( np.array(obs[0][3][9:21])) + list(np.array(obs[0][2][-3:-2])) + list(np.array(obs[0][3][-3:-2])) +
                            list(np.array(obs[0][2][-2:])) + list(np.array(obs[0][3][-2:])))
        share_obs_group2 = np.array(share_obs_group2)


        for agent_id in range(0, 2):
            if not self.use_centralized_V:
                share_obs_group1 = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs_group1.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

        for agent_id in range(2, 4):
            if not self.use_centralized_V:
                share_obs_group2 = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs_group2.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()


    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                # action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                action_env = action
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError


            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)


        # actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        share_obs_group1 = np.array(list(obs[0][0][0:3])+ list(obs[0][1][0:3]) + list(np.array(obs[0][0][3:21]))+
                            list(np.array(obs[0][1][9:21]))+ list(np.array(obs[0][0][-4:-2]))+
                            list(np.array(obs[0][0][-2:]))+list(np.array(obs[0][1][-2:])))
        share_obs_group1 = np.array(share_obs_group1)

        share_obs_group2 = np.array(list(obs[0][2][0:3]) + list(obs[0][3][0:3]) + list(np.array(obs[0][2][3:21]))
                            + list( np.array(obs[0][3][9:21])) + list(np.array(obs[0][2][-4:-2])) +
                            list(np.array(obs[0][2][-2:])) + list(np.array(obs[0][3][-2:])))
        share_obs_group2 = np.array(share_obs_group2)

        for agent_id in range(0, 2):
            if not self.use_centralized_V:
                share_obs_group1 = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].insert(
                share_obs_group1,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                np.array([actions])[:, agent_id][0].reshape(1, 4),
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

        for agent_id in range(2, 4):
            if not self.use_centralized_V:
                share_obs_group2 = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].insert(
                share_obs_group2,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                np.array([actions])[:, agent_id][0].reshape(1, 4),
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, episode, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset(0)

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        record_all_steps = []
        # print("net",[trainer.policy.actor for trainer in self.trainer])
        # print("net", [trainer.policy.critic for trainer in self.trainer])
        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = eval_action
#                    print('eval_action_env', eval_action_env)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Observe reward and next obs
            eval_obs, eval_rewards, eval_dones = self.eval_envs.step(eval_actions_env, eval_step)
            record_each_step = list(eval_actions_env) + list(eval_obs) + list(eval_rewards) + list(eval_rewards[0][0])
            record_all_steps = record_all_steps + record_each_step
            # save
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        " save the eval results in each 50 episodes"
        if episode % 50 == 0:
            result_record_name = '../eval_result/'
            result_record = (Path(result_record_name))
            if not result_record.exists():
                os.makedirs(str(result_record))
            result_record_name = '../eval_result/result_record' + str(episode) + '.csv'
            with open(result_record_name, 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(0, self.episode_length):
                    writer.writerow(record_all_steps[i * 4: (i + 1) * 4])

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
#                        print('action_env', action_env)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env, step)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
