"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time

class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        fixed_temperature=False,
        scheduler=None,
        device="cuda"
    ):
        self.model = model
        self.stochastic_policy = model.stochastic_policy
        self.optimizer = optimizer
        self.fixed_temperature = fixed_temperature
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        
        if self.stochastic_policy:
            self.train_iteration = self.stochastic_train_iteration
        else:
            self.train_iteration = self.deterministic_train_iteration
        
        self.start_time = time.time()
    
    def stochastic_train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        losses, nlls, state_losses, reward_losses = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, state_entropy, state_loss, reward_entropy, reward_loss, lr = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            state_losses.append(state_loss)
            reward_losses.append(reward_loss)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropy
        logs["training/state_entropy"] = state_entropy
        logs["training/state_loss"] = np.mean(state_losses)
        logs["training/reward_entropy"] = reward_entropy
        logs["training/reward_loss"] = reward_loss
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        logs["training/current_lr"] = lr[-1]

        return logs
    
    def deterministic_train_iteration(
        self,
        loss_fn,
        dataloader
    ):
        mse_loss, action_losses, state_losses, reward_losses = [], [], [], []
        logs = dict()
        train_start = time.time()

        for _, trajs in enumerate(dataloader):
            loss, action_loss, state_loss, reward_loss, lr = self.train_step_deter(loss_fn, trajs)
            mse_loss.append(loss)
            action_losses.append(action_loss)
            state_losses.append(state_loss)
            reward_losses.append(reward_loss)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(mse_loss)
        logs["training/train_loss_std"] = np.std(mse_loss)
        logs["training/action_loss"] = np.mean(action_losses)
        logs["training/state_loss"] = np.mean(state_losses)
        logs["training/reward_loss"] = np.mean(reward_losses)
        logs["training/current_lr"] = lr[-1]
        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)        
        
        action_target = torch.clone(actions)
        states_target = torch.clone(states)
        reward_target = torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy, state_entropy, state_loss, reward_entropy, reward_loss = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
            state_preds,
            states_target,
            reward_preds,
            reward_target
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        if not self.fixed_temperature:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (
                self.model.temperature() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            state_entropy.detach().cpu().item(),
            state_loss.detach().cpu().item(),
            reward_entropy.detach().cpu().item(),
            reward_loss.detach().cpu().item(),
            self.scheduler.get_last_lr()
        )

    def train_step_deter(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)        
        
        # get target values
        action_target = torch.clone(actions)
        state_target = torch.clone(states)
        reward_target = torch.clone(rewards)

        state_preds, action_preds, reward_pred = self.model.forward(
            states,
            actions,
            rewards,
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        # noisy targets
        # having twice the batch size, one batch of regular targets and one with noisy targets.
        # ~> results showed that its different than just adding half the noise to the targets on one batch
        noise = (torch.randn_like(action_target) * 0.125).clamp(-0.5, 0.5)
        action_target = action_target.clone().add(noise.to(self.device))
        # action_target = torch.concat([action_target, noisy_targets], dim=0)
        # action_preds = action_preds.repeat(2,1,1)
        
        # TODO: test with state_target noise and reward_target noise
        
        loss, action_loss, state_loss, reward_loss = loss_fn(action_preds, action_target,
                                                             state_preds, state_target,
                                                             reward_pred, reward_target,
                                                             padding_mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (loss.detach().cpu().item(),
                action_loss.detach().cpu().item(),
                state_loss.detach().cpu().item(),
                reward_loss.detach().cpu().item(), 
                self.scheduler.get_last_lr())
