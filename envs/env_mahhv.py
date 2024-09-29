import numpy as np

# single env
class MAHHVEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions, x):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions, x)
        return self.step_wait()

    def step_async(self, actions, x):
        self.actions = actions
        self.x = np.array([[x],[x],[x],[x]])

    def step_wait(self):
        results = [env.step(a, x) for (a, env, x) in zip(self.actions, self.envs, self.x)]
        obs, rews, dones = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones

    def reset(self, x):
        obs = [env.reset(x) for env in self.envs] # [env_num, agent_num, obs_dim]
#        print('ooobs', obs)
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError