# TODO: Make similar to random rollouts just using the DT ability to predict actions itself
# Stochastic action prediction where you sample the actions at each step
import torch
import numpy as np

from planner.base import MPC

class Dream(MPC):
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
        super(Dream, self).__init__(model=model,
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
        self.stochastic_policy = model.stochastic_policy
    
    def get_action(self, state: np.array)-> torch.Tensor:
        # repeat on batch dimension for number of parallel rollouts
        initial_states = state.repeat(self.parallel_rollouts, axis=1)
        trajectories = self.rollout(initial_states)
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
    
    def sample_predictions(self, action_pred, state_pred, reward_pred):
        if self.stochastic_policy:
            # the return action is a SquashNormal distribution
            action = action_pred.sample().reshape(self.parallel_rollouts, -1, self.action_dim)[:, -1]
            state = state_pred.sample().reshape(self.parallel_rollouts, -1, self.state_dim)[:, -1]
            state = state.reshape(self.parallel_rollouts, 1, self.state_dim)
            reward = reward_pred.sample().reshape(self.parallel_rollouts, -1, 1)[:, -1]
            # if use_mean:
            #     action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1] 
        else:
            action = action_pred.reshape(self.parallel_rollouts, -1, self.action_dim)[:, -1]
            noise = torch.normal(mean=torch.zeros_like(action), std=torch.ones_like(action) * self.action_high * 0.125).to(action.device)
            action = action + noise
            
            state = state_pred.detach().reshape(self.parallel_rollouts, 1, -1)
            reward = reward_pred

        action = action.clamp(self.action_low, self.action_high)
        return action, state, reward
    
    
    
    @torch.no_grad()
    def rollout(self, initial_state: np.array):
        self.model.eval()
        self.model.to(device=self.device)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = (
            torch.from_numpy(initial_state)
            .reshape(self.parallel_rollouts, self.state_dim)
            .to(device=self.device, dtype=torch.float32)
        ).reshape(self.parallel_rollouts, -1, self.state_dim) # (B, 1, F)
        
        actions = torch.zeros(0, device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        
        timesteps = torch.tensor([0] * self.parallel_rollouts,
                                 device=self.device,
                                 dtype=torch.long).reshape(self.parallel_rollouts, -1
        )
        episode_length = np.full(self.parallel_rollouts, np.inf)
        unfinished = np.ones(self.parallel_rollouts).astype(bool)      
        for t in range(self.rollout_horizon):

            # add padding
            actions = torch.cat(
                [
                    actions,
                    torch.zeros((self.parallel_rollouts, self.action_dim), device=self.device).reshape(
                        self.parallel_rollouts, -1, self.action_dim
                    ),
                ],
                dim=1,
            )
            rewards = torch.cat(
                [
                    rewards,
                    torch.zeros((self.parallel_rollouts, 1), device=self.device).reshape(self.parallel_rollouts, -1, 1),
                ],
                dim=1,
            )

            state_pred, action_pred, reward_pred = self.model.get_predictions(
                states=(states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
                actions=actions.to(dtype=torch.float32),
                rewards=torch.sign(rewards.to(dtype=torch.float32)) * torch.log(torch.abs(rewards.to(dtype=torch.float32)) + 1),
                timesteps=timesteps.to(dtype=torch.long),
                num_envs=self.parallel_rollouts,
            )
            # TODO: action sampling
            action, state, reward = self.sample_predictions(action_pred, state_pred, reward_pred)

            # Renormalize state and reward prediction
            renorm_state = (state * self.state_std) + self.state_mean
            renorm_reward = torch.sign(reward) * (torch.exp(torch.abs(reward)) - 1)
            
            # append predictions to rollout history
            # shapes [batch, time, feature]
            states = torch.cat([states, renorm_state], dim=1)
            rewards[:, -1] = renorm_reward
            actions[:, -1] = action
            
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
        
        trajectories = []
        for ii in range(self.parallel_rollouts):
            # ep_len = episode_length[ii].astype(int)
            # terminals = np.zeros(ep_len)
            # terminals[-1] = 1
            traj = {
                "observations": states[ii].detach().cpu().numpy()[:self.rollout_horizon], # to get rid of possible last next state
                "actions": actions[ii].detach().cpu().numpy()[:self.rollout_horizon],
                "rewards": rewards[ii].detach().cpu().numpy()[:self.rollout_horizon],
                # "terminals": terminals,
            }
            trajectories.append(traj)
        return trajectories