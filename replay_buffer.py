"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, capacity, adding_type="ffo", trajectories=[]):
        self.capacity = capacity
        self.adding_type = adding_type
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)
    
    def best_traj_stats(self, best_x=10):
        returns = np.array([sum(traj["rewards"]) for traj in self.trajectories])
        lengths = np.array([len(traj["rewards"]) for traj in self.trajectories])
        best_returns = np.sort(returns).squeeze()[-best_x:]  # lowest to highest
        best_lengths = np.sort(lengths).squeeze()[-best_x:]  # lowest to highest
        return np.mean(best_returns), np.std(best_returns), np.mean(best_lengths), np.min(returns)
    
    def buffer_stats(self, ):
        returns = np.array([sum(traj["rewards"]) for traj in self.trajectories])
        lengths = np.array([len(traj["rewards"]) for traj in self.trajectories])
        max_return, min_return, mean_return = np.max(returns), np.min(returns), np.mean(returns)
        max_traj_lenghts, min_traj_length, mean_traj_length = np.max(lengths), np.min(lengths), np.mean(lengths)
        
        norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-12)
        p_sample = torch.softmax(torch.from_numpy(norm_returns).float(), dim=0).squeeze().numpy()
        
        # torch_returns = torch.from_numpy(returns).float()
        # p_sample = torch.softmax(torch_returns, dim=0).squeeze().numpy()
        
        # torch_returns = torch.from_numpy(returns).float()
        # p_sample = torch.softmax(torch_returns / torch.sqrt(torch.FloatTensor([len(returns)])), dim=0).squeeze().numpy()
        
        # p_sample = (returns / np.sum(returns)).squeeze()
        
        # when sampling for traj length
        # traj_lens = np.array([len(traj["observations"]) for traj in self.trajectories])
        # p_sample = traj_lens / np.sum(traj_lens)
        
        output = {"buffer/max_return": max_return,
                  "buffer/min_return": min_return,
                  "buffer/mean_return": mean_return,
                  "buffer/max_traj_len": max_traj_lenghts,
                  "buffer/min_traj_len": min_traj_length,
                  "buffer/mean_traj_len": mean_traj_length,
                  "buffer/p_sample": p_sample,}
        return output

    def add_new_trajs(self, new_trajs, prefill=False):
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        else:
            if self.adding_type == "ffo":
                self.trajectories[
                    self.start_idx : self.start_idx + len(new_trajs)
                ] = new_trajs
                self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity
            elif self.adding_type == "return":
                returns = np.array([sum(traj["rewards"]) for traj in self.trajectories])
                return_idxs = np.argsort(returns.squeeze())  # lowest to highest
                num_new_traj = len(new_trajs)
                traj_2_replace = return_idxs[:num_new_traj]
                for (idx, new_traj) in zip(traj_2_replace, new_trajs):
                    self.trajectories[idx] = new_traj
                
            elif self.adding_type == "traj_len":
                lengths = np.array([len(traj["rewards"]) for traj in self.trajectories])
                lengths_idxs = np.argsort(lengths).squeeze()  # lowest to highest
                num_new_traj = len(new_trajs)
                traj_2_replace = lengths_idxs[:num_new_traj]
                for (idx, new_traj) in zip(traj_2_replace, new_trajs):
                    self.trajectories[idx] = new_traj
            else:
                raise NotImplementedError

        assert len(self.trajectories) <= self.capacity
