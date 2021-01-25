from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
from .stag_hunt import StagHunt

REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

import multiagent
from .MAPredatorPreyWrappers import RandomPreyActions,PredatorPreyTerminator
import gym

def env_fn2( **kwargs):
    print(kwargs)
    env = gym.make(kwargs.get('env_args').get("name"))
    env = RandomPreyActions(env)
    env = PredatorPreyTerminator(env)
    return env

REGISTRY["predator_prey"] = partial(env_fn2)

from .ctf.ctf import CTF, CTF_v2

REGISTRY["ctf"] = partial(env_fn, env=CTF)
REGISTRY["ctf2"] = partial(env_fn, env=CTF_v2)
