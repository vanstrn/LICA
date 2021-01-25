
import configparser
import numpy as np
import random
from utils.dict2namedtuple import convert
from envs.multiagentenv import MultiAgentEnv
import gym, gym_cap
import gym_cap.heuristic as policy

int_type = np.int16
float_type = np.float32


class CTF(MultiAgentEnv):

    action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'stay': 4, 'catch': 5,
                     'look-right': 6, 'look-down': 7, 'look-left': 8, 'look-up': 9}
    action_look_to_act = 6

    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        # if isinstance(args, dict):
        #     args = convert(args)
        self.args = args

        self.capture_action = getattr(args, "capture_action", False)

        map_size = args.pop("map_size")
        nchannels = 6
        # exit()
        args.pop("seed")
        self.game_config = configparser.ConfigParser()
        for k,v in args.items():
            self.game_config[k] = v

        #Building the CTF environment.
        self.env = gym.make("cap-v0",map_size=map_size, config_path=self.game_config)

        self.state_size = [nchannels,map_size,map_size]
        self.obs_size = [nchannels,map_size*2-1,map_size*2-1]

        self.env.reset(config_path=self.game_config, policy_red=policy.Roomba())
        # Define the agents and their action space
        self.n_actions = 5
        self.n_agents = len(self.env.get_team_blue)
        self.episode_limit = args["control"]["MAX_STEP"]

        # self.agent_obs = args.agent_obs
        # self.agent_obs_dim = np.asarray(self.agent_obs, dtype=int_type)



    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------
    def reset(self):
        # Reset old episode
        state = self.env.reset(config_path=self.game_config, policy_red=policy.Roomba())
        # self.step(th.zeros(self.n_agents).fill_(self.action_labels['stay']))
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Execute a*bs actions in the environment. """
        _,r,terminated,_ = self.env.step(actions)

        info = {
            "win_rate":self.env.blue_win,
        }

        return r, int(terminated), info

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        #Centering state on specific agent.
        padder=[0,0,0,1,0,0]
        #Get list of controlled agents
        s0 = self.env.get_obs_blue.astype(np.float32)

        olx, oly, ch = s0.shape
        H = olx*2-1
        W = oly*2-1
        padder = padder[:ch]

        cx, cy = (W-1)//2, (H-1)//2
        states = np.zeros([ H, W, len(padder)])
        states[:,:,:] = np.array(padder)
        x, y = self.env.get_team_blue[agent_id].get_loc()
        states[max(cx-x,0):min(cx-x+olx,W),max(cy-y,0):min(cy-y+oly,H),:] = s0
        return np.swapaxes(states,0,2)

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        # Either return the state as a list of entities...
        # ... or return the entire grid
        return np.swapaxes(self.env.get_obs_blue.astype(np.float32),0,2)

    def get_obs_intersect_pair_size(self):
        return 2 * self.get_obs_size()

    def get_obs_intersect_all_size(self):
        return self.n_agents * self.get_obs_size()

    def get_obs_intersection(self, agent_ids):
        return self._observe(agent_ids)

    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        """ Currently runs only with batch_size==1. """
        return [1]*self.n_actions

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_stats(self):
        pass

    def get_env_info(self):
        info = MultiAgentEnv.get_env_info(self)
        return info

    # --------- RENDER METHODS -----------------------------------------------------------------------------------------
    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing Multi-Agent Navigation")

    def render_array(self):
        # Return an rgb array of the frame
        return None

    def render(self):
        # TODO!
        self.env.render()

    def seed(self):
        raise NotImplementedError


class CTF_v2(CTF):

    action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'stay': 4, 'catch': 5,
                     'look-right': 6, 'look-down': 7, 'look-left': 8, 'look-up': 9}
    action_look_to_act = 6

    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        # if isinstance(args, dict):
        #     args = convert(args)
        self.args = args
        args.pop("seed")

        map_size = args.pop("map_size")
        nchannels = 6
        self.game_config = configparser.ConfigParser()
        for k,v in args.items():
            self.game_config[k] = v


        self.capture_action = getattr(args, "capture_action", False)

        # exit()

        #Building the CTF environment.
        self.env = gym.make("cap-v0",map_size=map_size, config_path=self.game_config)

        self.state_size = map_size*map_size*nchannels
        self.obs_size = (map_size*2-1)*(map_size*2-1)*nchannels

        # Define the agents and their action space

        state = self.env.reset(config_path=self.game_config, policy_red=policy.Roomba())

        self.n_actions = 5
        self.n_agents = len(self.env.get_team_blue)
        self.episode_limit = args["control"]["MAX_STEP"]

        # self.agent_obs = args.agent_obs
        # self.agent_obs_dim = np.asarray(self.agent_obs, dtype=int_type)

        self.reset()


    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        #Centering state on specific agent.
        padder=[0,0,0,1,0,0]
        #Get list of controlled agents
        s0 = self.env.get_obs_blue.astype(np.float32)

        olx, oly, ch = s0.shape
        H = olx*2-1
        W = oly*2-1
        padder = padder[:ch]

        cx, cy = (W-1)//2, (H-1)//2
        states = np.zeros([1, H, W, len(padder)])
        states[:,:,:] = np.array(padder)
        x, y = self.env.get_team_blue[agent_id].get_loc()
        states[0,max(cx-x,0):min(cx-x+olx,W),max(cy-y,0):min(cy-y+oly,H),:] = s0
        return states.flatten()

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        # Either return the state as a list of entities...
        # ... or return the entire grid

        return self.env.get_obs_blue.astype(np.float32).flatten()
