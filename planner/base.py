import torch
import numpy as np

class MPC():
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
                 reward_scale=0.001,
                 device="cpu")-> None:

        self.model = model
        self.device = device
        self.action_dim = action_dim
        self.action_low = min(action_range)
        self.action_high = max(action_range)
        
        self.state_dim = state_dim
        self.state_mean = torch.from_numpy(state_mean).to(device=device)
        self.state_std = torch.from_numpy(state_std).to(device=device)
        
        self.reward_scale = reward_scale

        self.parallel_rollouts = parallel_rollouts
        self.rollout_horizon = rollout_horizon
        self.max_ep_len = max_ep_len
    
    def get_action(self, state: torch.Tensor)-> torch.Tensor:
        raise NotImplementedError
    
    def extract_best_action(self, traj: tuple):
        states, actions, rewards = traj
        # with shape (batch, time, feature)
        returns = rewards.sum(1)
        best_traj = torch.argmax(returns)
        best_action = actions[best_traj, 0, :]
        return best_action.cpu().numpy()
    
    @torch.no_grad()
    def rollout(self, initial_state: np.array, action_candidates: torch.Tensor):
        self.model.eval()
        self.model.to(device=self.device)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = (
            torch.from_numpy(initial_state)
            .reshape(self.parallel_rollouts, self.state_dim)
            .to(device=self.device, dtype=torch.float32)
        ).reshape(self.parallel_rollouts, -1, self.state_dim) # (B, 1, F)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        
        timesteps = torch.tensor([0] * self.parallel_rollouts,
                                 device=self.device,
                                 dtype=torch.long).reshape(self.parallel_rollouts, -1
        )
        episode_length = np.full(self.parallel_rollouts, np.inf)
        unfinished = np.ones(self.parallel_rollouts).astype(bool)      
        for t in range(self.rollout_horizon):
            # add action candidate to rollout history
            actions = action_candidates[:, :t+1, :]
            # actions = torch.cat(
            #     [
            #         actions,
            #         action_candidates[:, t, :],
            #     ],
            #     dim=1,
            # )
            # add reward padding
            rewards = torch.cat(
                [
                    rewards,
                    torch.zeros((self.parallel_rollouts, 1), device=self.device).reshape(self.parallel_rollouts, -1, 1),
                ],
                dim=1,
            )

            state_pred, _, reward_pred = self.model.get_predictions_reward(
                states=(states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
                actions=actions.to(dtype=torch.float32),
                rewards=torch.sign(rewards.to(dtype=torch.float32)) * torch.log(torch.abs(rewards.to(dtype=torch.float32)) + 1),
                timesteps=timesteps.to(dtype=torch.long),
                num_envs=self.parallel_rollouts,
            )
            state_pred = state_pred.detach().reshape(self.parallel_rollouts, 1, -1)
            # Renormalize state and reward prediction
            renorm_state_pred = (state_pred * self.state_std) + self.state_mean
            renorm_reward_pred = torch.sign(reward_pred) * (torch.exp(torch.abs(reward_pred)) - 1)
            
            # append predictions to rollout history
            # shapes [batch, time, feature]
            states = torch.cat([states, renorm_state_pred], dim=1)
            rewards[:, -1] = renorm_reward_pred
            
            
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((self.parallel_rollouts, 1), device=self.device, dtype=torch.long).reshape(
                        self.parallel_rollouts, 1
                    )
                    * (t + 1),
                ],
                dim=1,
            )
            
            # TODO: do we need terminal prediction?
            # if t == self.max_ep_len - 1:
            #     done = np.ones(done.shape).astype(bool)

            # if np.any(done):
            #     ind = np.where(done)[0]
            #     unfinished[ind] = False
            #     episode_length[ind] = np.minimum(episode_length[ind], t + 1)

            # if not np.any(unfinished):
            #     break
        
        # trajectories = []
        # for ii in range(self.parallel_rollouts):
        #     # ep_len = episode_length[ii].astype(int)
        #     # terminals = np.zeros(ep_len)
        #     # terminals[-1] = 1
        #     traj = {
        #         "observations": states[ii].detach().cpu().numpy()[:self.rollout_horizon], # to get rid of possible last next state
        #         "actions": actions[ii].detach().cpu().numpy()[:self.rollout_horizon],
        #         "rewards": rewards[ii].detach().cpu().numpy()[:self.rollout_horizon],
        #         # "terminals": terminals,
        #     }
        #     trajectories.append(traj)
        return (states, actions, rewards)