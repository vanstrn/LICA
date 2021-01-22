
import gym
from gym import spaces
import numpy as np


class RandomPreyActions(gym.core.Wrapper):
    """
    Wrapper wrap observation into dictionary
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_limit =200
        self.observation_space = self.observation_space[0].shape
        self.state_space = self.state_space.shape
        self.action_space_prey = self.action_space[-1]
        self.action_space = self.action_space[0].n
        self.n_agents=self.env.n-1


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation.pop(-1)
        self.obs = observation
        return observation

    def step(self, action):
        actionPrey = self.action_space_prey.sample()
        action_list = list(action)
        action_list.append(actionPrey)
        observation, reward, done, info = self.env.step(action=action_list)
        observation.pop(-1)
        reward.pop(-1)
        if isinstance(done,list):
            done.pop(-1)

        self.obs = observation

        return reward, done, info

    def get_env_info(self):
        info = {"state_shape": self.state_space[0],
                "obs_shape": self.observation_space[0],
                "n_actions": self.action_space,
                "n_agents": self.n_agents,
                "episode_limit": 1000}
        return info

    def get_avail_actions(self):
        return [np.ones(self.action_space) for i in range(self.n_agents)]
    def get_obs(self):
        return self.obs



class PredatorPreyTerminator(gym.core.Wrapper):
    """
    Processes the tracked data of the environment.
    In this case it sums the reward over the entire episode.
    """
    def __init__(self,env, **kwargs):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done, info = self.env.step(action=action)
        if reward[0]>0:
            return  reward[0], True, {}
        elif isinstance(done,bool):
            return  reward[0], done, {}
        else:
            return  reward[0], done[0], {}
