from planner.base import MPC
import torch
import numpy as np

class RandomShooting(MPC):
    def __init__(self,
                 model,
                 action_dim,
                 action_range,
                 state_dim,
                 state_mean,
                 state_std,
                 parallel_rollouts,
                 rollout_horizon,
                 max_ep_len=1000,
                 reward_scale=0.001, # TODO Not used yet! symlog?
                 device="cpu") -> None:
        super(RandomShooting, self).__init__(model=model,
                                             action_dim=action_dim,
                                             action_range=action_range,
                                             state_dim=state_dim,
                                             state_mean=state_mean,
                                             state_std=state_std,
                                             parallel_rollouts=parallel_rollouts,
                                             rollout_horizon=rollout_horizon,
                                             max_ep_len=max_ep_len,
                                             reward_scale=reward_scale,
                                             device=device)


    def _sample_action(self, )-> torch.Tensor:
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.parallel_rollouts,
                                          self.rollout_horizon,
                                          self.action_dim))

        return torch.from_numpy(actions).to(self.device).float()
    
    def get_action(self, state: np.array)-> torch.Tensor:
        # repeat on batch dimension for number of parallel rollouts
        initial_states = state.repeat(self.parallel_rollouts, axis=1)
        rollout_action_candidates = self._sample_action()
        trajectories = self.rollout(initial_states, rollout_action_candidates)
        best_traj_action = self.extract_best_trajectory(trajectories)
        return best_traj_action

    def extract_best_trajectory(self, trajs: dict):
        returns = []
        for traj in trajs:
            for k, v in traj.items():
                if k == "rewards":
                    returns.append(np.sum(v))
        best_action_traj_idx = np.argmax(returns)
        return trajs[best_action_traj_idx]["actions"][0]
