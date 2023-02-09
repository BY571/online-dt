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

        self.masked_actions = torch.zeros((self.parallel_rollouts, 1, self.action_dim),
                                           device=self.device, dtype=torch.float32)
        self.masked_rewards = torch.zeros((self.parallel_rollouts, 1, 1),
                                           device=self.device, dtype=torch.float32)
        self.timestep_adder = torch.ones((self.parallel_rollouts, 1),
                                          device=self.device, dtype=torch.long)

    def get_action(self, state: np.array)-> torch.Tensor:
        # repeat on batch dimension for number of parallel rollouts
        initial_states = state.repeat(self.parallel_rollouts, axis=1)
        trajectories = self.rollout(initial_states)
        best_traj_action = self.extract_best_action(trajectories)
        return best_traj_action

    def sample_predictions(self, action_pred, state_pred, reward_pred):
        if self.stochastic_policy:
            # the return action is a SquashNormal distribution
            action = action_pred.sample[:, -1] # (batch, feature)
            state = (state_pred.sample()[:, -1,:]).view(self.parallel_rollouts, 1, self.state_dim)
            reward = reward_pred.sample()[:, -1] # (batch, 1)
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
        states = torch.from_numpy(initial_state).view(self.parallel_rollouts, 1, self.state_dim).to(device=self.device, dtype=torch.float32)
         # (B, 1, F)
        
        actions = torch.zeros(0, device=self.device, dtype=torch.float32)       
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

        timesteps = torch.zeros((self.parallel_rollouts, 1),
                                device=self.device,
                                dtype=torch.long)
        
        #episode_length = np.full(self.parallel_rollouts, np.inf)
        #unfinished = np.ones(self.parallel_rollouts).astype(bool)      
        for t in range(self.rollout_horizon):

            # add padding
            actions = torch.cat([actions, self.masked_actions], dim=1)
            # (batch, t+1, feature)
            rewards = torch.cat(
                [rewards, self.masked_rewards],dim=1)
            # (batch, t+1, feature)

            state_pred, action_pred, reward_pred = self.model.get_predictions(
                states=(states - self.state_mean) / self.state_std, # normalize
                actions=actions,
                rewards=torch.sign(rewards) * torch.log(torch.abs(rewards) + 1), # scale
                timesteps=timesteps,
                num_envs=self.parallel_rollouts,
            )

            action, state, reward = self.sample_predictions(action_pred, state_pred, reward_pred)

            # Renormalize state and reward prediction
            renorm_state = (state * self.state_std) + self.state_mean
            renorm_reward = torch.sign(reward) * (torch.exp(torch.abs(reward)) - 1)
            
            # append predictions to rollout history
            # shapes [batch, time, feature]
            states = torch.cat([states, renorm_state], dim=1)
            rewards[:, -1] = renorm_reward
            actions[:, -1] = action
            
            timesteps = torch.cat([timesteps, self.timestep_adder * (t + 1),], dim=1)
            
            # TODO: do we need terminal prediction?
            # if t == self.max_ep_len - 1:
            #     done = np.ones(done.shape).astype(bool)

            # if np.any(done):
            #     ind = np.where(done)[0]
            #     unfinished[ind] = False
            #     episode_length[ind] = np.minimum(episode_length[ind], t + 1)

            # if not np.any(unfinished):
            #     break
        
        return (states, actions, rewards)