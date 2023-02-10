"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch

MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    vec_env,
    state_dim,
    act_dim,
    action_range,
    state_mean,
    state_std,
    device,
    stochastic_policy=True,
    use_mean=False,
    exploration_noise=0.1,
    noise_sample_inter=1,
):
    def eval_episodes_fn(model):
        returns, lengths, _ = vec_evaluate_episode(
            vec_env,
            state_dim,
            act_dim,
            action_range,
            model,
            max_ep_len=MAX_EPISODE_LEN,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            stochastic_policy=stochastic_policy,
            use_mean=True,
            exploration_noise=0.0,
            noise_sample_inter=noise_sample_inter,
        )
        return {
            f"evaluation/return_mean": np.mean(returns),
            f"evaluation/return_std": np.std(returns),
            f"evaluation/length_mean": np.mean(lengths),
            f"evaluation/length_std": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode(
    vec_env,
    state_dim,
    act_dim,
    action_range,
    model,
    max_ep_len=1000,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    stochastic_policy=True,
    use_mean=False,
    exploration_noise=0.0,
    noise_sample_inter=1,
):

    model.eval()
    model.to(device=device)

    # TODO: take those off for symlog
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).view(num_envs, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    action_mask = torch.zeros((num_envs, 1, act_dim), device=device)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    reward_mask = torch.zeros((num_envs, 1, 1), device=device)

    timesteps = torch.zeros((num_envs, 1), device=device, dtype=torch.long)
    timestep_mask = torch.ones((num_envs, 1), device=device, dtype=torch.long)

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, action_mask], dim=1)
        rewards = torch.cat([rewards, reward_mask], dim=1)

        _, action_dist, _ = model.get_predictions(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            # torch.sign(states.to(dtype=torch.float32))*torch.log(torch.abs(states.to(dtype=torch.float32)) + 1),
            actions=actions.to(dtype=torch.float32),
            rewards=torch.sign(rewards.to(dtype=torch.float32))*torch.log(torch.abs(rewards.to(dtype=torch.float32)) + 1), # rewards.to(dtype=torch.float32),
            timesteps=timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        
        if stochastic_policy:
            # state reward predictions
            # state_pred = state_pred.sample().reshape(num_envs, -1, state_dim)[:, -1]
            # state_pred = state_pred.detach().cpu().numpy()
            # reward_pred = reward_pred.sample().reshape(num_envs, -1, 1)[:, -1]
            # reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
            # the return action is a SquashNormal distribution
            action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
            # if t % noise_sample_inter == 0:
            #     noise = torch.normal(mean=torch.zeros_like(action), std=torch.ones_like(action) * action_range[1] * exploration_noise).to(action.device)
            # action = action + noise
            # added state sampling
            #state_pred = state_pred.sample().reshape(num_envs, -1, state_dim)[:, -1]
            #
            if use_mean:
                action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
                # TODO: solve this reshape issue
                # state_pred = state_pred.mean.reshape(num_envs, -1, state_dim)[:, -1]
                #state_pred = state_pred.sample().reshape(num_envs, -1, state_dim)[:, -1]
                
            
        else:
            # state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
            # reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
            action = action_dist.reshape(num_envs, -1, act_dim)[:, -1]
            if not use_mean:
                if t % noise_sample_inter == 0:
                    noise = torch.normal(mean=torch.zeros_like(action), std=torch.ones_like(action) * action_range[1] * exploration_noise).to(action.device)
                action = action + noise

        action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        timesteps = torch.cat([timesteps, timestep_mask * (t + 1)], dim=1)

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    trajectories = []
    ep_lengths = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        ep_lengths.append(ep_len)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        ep_lengths,
        trajectories,
    )


@torch.no_grad()
def planned_augment(
    vec_env,
    planner,
    max_ep_len=1000,
):
    # TODO: currently only supports single environment!
    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    states, actions, rewards, done_states = [], [], [], []
    for t in range(max_ep_len):

        action = planner.get_action(state)

        next_state, reward, done, _ = vec_env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        done_states.append(done)
        
        state = next_state
        
        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    traj = {
        "observations": np.stack(states).squeeze(), # ugly squeeze
        "actions": np.stack(actions).astype(dtype="float32"), # (1000, action_dim) float32
        "rewards": np.stack(rewards).astype(dtype="float32"), # (1000, 1) float32 
        "terminals": np.stack(done_states).astype(dtype=float).squeeze() # (1000, )
    }
    episode_return = np.sum(rewards, keepdims=True).reshape(-1)
    return (
        episode_return,
        episode_length.reshape(num_envs),
        [traj],
    )


def random_collect(vec_env,
                   state_dim,
                   act_dim,
                   max_ep_len=1000):
    num_envs = vec_env.num_envs
    state = vec_env.reset()

    states = (state.reshape(num_envs, state_dim)).reshape(num_envs, -1, state_dim)
    actions = np.zeros((num_envs, act_dim)).reshape(num_envs, -1, act_dim)
    action_mask = np.zeros((num_envs, 1, act_dim))
    rewards = np.zeros((num_envs, 1)).reshape(num_envs, -1, 1)
    reward_mask = np.zeros((num_envs, 1)).reshape(num_envs, -1, 1)
    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = np.concatenate([actions, action_mask], axis=1)
        rewards = np.concatenate([rewards, reward_mask], axis=1)
        action = vec_env.action_space.sample()
        state, reward, done, _ = vec_env.step(action)
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
        actions[:, -1] = action
        state = state.reshape(num_envs, -1, state_dim)
        states = np.concatenate([states, state], axis=1)
        reward = np.array(reward).reshape(num_envs, 1)
        rewards[:, -1] = reward
         
        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break
    trajectories = []

    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii][:ep_len],
            "actions": actions[ii][:ep_len],
            "rewards": rewards[ii][:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )
        
@torch.no_grad()
def video_evaluate_episode_rtg(
    env,
    vid,
    state_dim,
    act_dim,
    action_range,
    model,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    stochastic_policy=True,
    use_mean=False,
    exploration_noise=1.0,
    noise_window=5,
):

    model.eval()
    model.to(device=device)

    # Take those off for symlog
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = 1
    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std, # symlog?
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
        if stochastic_policy:
            # the return action is a SquashNormal distribution
            action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
            if use_mean:
                action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        else:
            action = action_dist.reshape(num_envs, -1, act_dim)[:, -1]
            if not use_mean:
                if t % noise_window == 0:
                    noise = torch.normal(mean=torch.zeros_like(action), std=torch.ones_like(action) * action_range[1] * exploration_noise).to(action.device)
                action = action + noise

        action = action.clamp(*model.action_range)
        vid.capture_frame()
        state, reward, done, _ = env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(np.array([reward])).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(1).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)
    print(5*"-" + " Evaluation Result " + 5 * "-")
    print("Trajectory Return: ", episode_return.item())
    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )
    vid.close()