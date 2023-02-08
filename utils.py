"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import random
import torch
import numpy as np
from pathlib import Path
import gym

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_np(t):
    """
    convert a torch tensor to a numpy array
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_env_builder(seed, env_name, target_goal=None):
    def make_env_fn():
        print(f"*** Creating Environment: {env_name} ***")
        env = gym.make(env_name)# , healthy_reward=0.75)
        env.seed(seed)
        if hasattr(env.env, "wrapped_env"):
            env.env.wrapped_env.seed(seed)
        elif hasattr(env.env, "seed"):
            env.env.seed(seed)
        else:
            pass
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        if target_goal:
            env.set_target_goal(target_goal)
            print(f"Set the target goal to be {env.target_goal}")
        return env

    return make_env_fn