"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
from exploration.rnd.default import RndPredictor

class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
        use_rnd=False,
        rnd_pred=None,
        rnd_trg=None,
        rnd_opti=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_rnd = use_rnd
        self.rnd_pred = rnd_pred
        self.rnd_trg = rnd_trg
        self.rnd_opti = rnd_opti
        
        self.start_time = time.time()
    

    def train_rnd(self, states):
        bs = states.shape[0]
        state_dim = states.shape[1]
        states = states.reshape(bs*state_dim, -1)
        state_pred = self.rnd_pred(states)
        with torch.no_grad():
            random_targets = self.rnd_trg(states)
        state_pred = state_pred.reshape(bs, state_dim, -1)
        random_targets = random_targets.reshape(bs, state_dim, -1)
        pred_error = ((state_pred - random_targets)**2).mean(dim=-1, keepdim=True)
        rnd_loss = pred_error.mean()
        self.rnd_opti.zero_grad()
        rnd_loss.backward()
        self.rnd_opti.step()
        return rnd_loss.detach().cpu().mean().item()

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        losses, nlls, entropies, intrinsic_rewards = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, intrinsic_reward = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            intrinsic_rewards.append(intrinsic_reward)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        if self.use_rnd:
            logs["training/intrinsic_reward"] = np.mean(intrinsic_rewards)

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
        
        if self.use_rnd:
            rnd_loss = self.train_rnd(states)
        else:
            rnd_loss = 0.0
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

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
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
            rnd_loss
        )
