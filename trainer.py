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
        scheduler=None,
        device="cuda"
    ):
        self.model = model
        self.stochastic_policy = model.stochastic_policy
        self.optimizer = optimizer
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

        losses, nlls, entropies, state_losses, state_entropies = [], [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, state_entropy, state_loss, lr = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            state_losses.append(state_loss)
            state_entropies.append(state_entropy)


        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/state_entropy"] = state_entropies[-1]
        logs["training/state_loss"] = np.mean(state_losses)
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        logs["training/current_lr"] = lr[-1]

        return logs
    
    def deterministic_train_iteration(
        self,
        loss_fn,
        dataloader
    ):
        mse_loss = []
        logs = dict()
        train_start = time.time()

        for _, trajs in enumerate(dataloader):
            loss, lr = self.train_step_deter(loss_fn, trajs)
            mse_loss.append(loss)
        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(mse_loss)
        logs["training/train_loss_std"] = np.std(mse_loss)
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

        state_preds, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        #noise = (torch.randn_like(action_target) * 0.01).clamp(-0.05, 0.05)
        #action_target = torch.tanh(action_target + noise.to(self.device))

        loss, nll, entropy, state_entropy, state_loss = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
            states_target,
            state_preds
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

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
            state_entropy,# .detach().cpu().item(),
            state_loss,#.detach().cpu().item()
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
        
        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        # noisy targets
        # having twice the batch size, one batch of regular targets and one with noisy targets.
        # ~> results showed that its different than just adding half the noise to the targets on one batch
        noise = (torch.randn_like(action_target) * 0.1).clamp(-0.5, 0.5)
        noisy_targets = action_target.clone().add(noise.to(self.device))
        action_target = torch.concat([action_target, noisy_targets], dim=0)
        action_preds = action_preds.repeat(2,1,1)
        
        loss = loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.detach().cpu().item(), self.scheduler.get_last_lr()