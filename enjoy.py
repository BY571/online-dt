"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

# from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers import Monitor
import d4rl
from more_itertools import sample
import torch
import numpy as np
import wandb

import utils

from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import video_evaluate_episode_rtg
from logger import WandbLogger

MAX_EPISODE_LEN = 1000


class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=variant["stochastic_policy"],
            ordering=variant["ordering"],
            init_temperature=0.0,
            target_entropy=self.target_entropy,
        ).to(device=self.device)
        self._load_model(variant["model_path"])
        self.stochastic_policy = variant["stochastic_policy"]

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        
    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range


    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std


    def evaluate(self, eval_fns, updated_r2g=None):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model, updated_r2g)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        # import d4rl
        if self.stochastic_policy:
            def loss_fn(
                a_hat_dist,
                a,
                attention_mask,
                entropy_reg,
            ):
                # a_hat is a SquashedNormal Distribution
                log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

                entropy = a_hat_dist.entropy().mean()
                loss = -(log_likelihood + entropy_reg * entropy)

                return (
                    loss,
                    -log_likelihood,
                    entropy,
                )
        else:
            loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                # import d4rl

                env = gym.make(env_name)
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

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None

        env = gym.make(env_name)
        env = Monitor(env, './dt-eval/video',video_callable=lambda episode_id: True, force=True)
        self.start_time = time.time()
        with wandb.init(dir="./dt-eval",
                        config=self.variant,
                        project="online-dt",
                        name=self.variant["exp_name"],
                        notes="video_evaluation",
                        group=self.variant["env"],
                        monitor_gym=True):

            video_evaluate_episode_rtg(env,
                                       state_dim=self.state_dim,
                                       act_dim=self.act_dim,
                                       action_range=self.action_range,
                                       model=self.model,
                                       target_return=-1,
                                       max_ep_len=1000,
                                       reward_scale=self.reward_scale,
                                       state_mean=self.state_mean,
                                       state_std=self.state_std,
                                       device="cuda",
                                       mode="normal",
                                       stochastic_policy=self.stochastic_policy,
                                       use_mean=True,
                                       exploration_noise=0.0)
            

            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    parser.add_argument("--stochastic_policy", type=int, choices=[0,1], default=1)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    parser.add_argument("--model_path", type=str, default="saved_model")

    # environment options
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
