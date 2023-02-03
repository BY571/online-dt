import argparse
import pickle
import random
import time
import gym

import torch
import numpy as np
import wandb

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg, random_collect
from trainer import SequenceTrainer
from logger import WandbLogger
from collections import deque

# try:
#     import gym_cartpole_swingup
# except:
#     raise ImportError


MAX_EPISODE_LEN = 1000


def get_offline_env(name="halfcheetah"):
    import d4rl
    if name == "HalfCheetah-v3" or name == "HalfCheetah-v2":
        return "halfcheetah-medium-v2"
    elif name == "Hopper-v3" or name == "Hopper-v2":
        return "hopper-medium-v2"
    # TODO: add other offline RL examples
    else:
        raise NotImplementedError

class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        
        # initialize by offline trajs
        if variant["expert_experience"]:
            self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(get_offline_env(variant["env"]))
            self.replay_buffer = ReplayBuffer(variant["replay_size"], adding_type=variant["buffer_adding"], trajectories=self.offline_trajs)
        else:
            self.state_mean, self.state_std = self._get_online_stats(variant)
            self.replay_buffer = ReplayBuffer(variant["replay_size"], adding_type=variant["buffer_adding"], trajectories=[])
        self.aug_trajs = []

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
            use_rtg=variant["use_rtg"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.stochastic_policy = variant["stochastic_policy"]
        self.exploration_noise = variant["exploration_noise"]
        self.noise_sample_inter = variant["noise_sample_inter"]
        

        if not variant["use_cosineanneal"]:
            self.optimizer = Lamb(
                self.model.parameters(),
                lr=variant["learning_rate"],
                weight_decay=variant["weight_decay"],
                eps=1e-8,
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.optimizer = Lamb(
                self.model.parameters(),
                lr=variant["learning_rate_reset"],
                weight_decay=variant["weight_decay"],
                eps=1e-8,
            )
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=variant["learning_rate_reset"], eps=1e-8, weight_decay=variant["weight_decay"]) #, eps=1e-8,
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 
                                            T_0=variant["lr_restart_updates"],# Number of iterations for the first restart
                                            T_mult=variant["decay_lr_factor"], # A factor increases T after a restart
                                            eta_min=variant["learning_rate"]) # Minimum learning rate

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        self.best_x_buffer = variant["best_x"]      
        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = WandbLogger(variant)
        
        self.running_buffer_mean = deque(maxlen=10)
        self.deter_iter = 0


    def _get_online_stats(self, variant):
        if "HalfCheetah" in variant["env"] or "Hopper" in variant["env"]:
            _, state_mean, state_std = self._load_dataset(get_offline_env(variant["env"]))
        else:
            env = gym.make(variant["env"])
            states = []
            states.append(env.reset())
            for i in range(1000):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)
                states.append(state)
                if done:
                    env.reset()
                    
            states = np.stack(states)
            state_mean = np.mean(states, axis=0)
            state_std = np.std(states, axis=0)
            env.close()
        return state_mean, state_std

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

    def _save_model(self, path_prefix, iteration=0, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model_{iteration}.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model_{iteration}.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model_{iteration}.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model_{iteration}.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
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

    
    def prefill_buffer(self, online_envs):
        """
        Collects trajectories with a random policy and adds them to the replay buffer
        """
        returns, lengths, trajs = random_collect(online_envs,
                                                 self.state_dim,
                                                 self.act_dim,
                                                 max_ep_len=MAX_EPISODE_LEN)
        self.replay_buffer.add_new_trajs(trajs, prefill=True)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)


    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN
        returns = -np.inf
        lengths = 0
        mean_returns = -np.inf
        mean_traj_lens = 1

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs
            mean_returns, std_returns, mean_traj_lens, min_return = self.replay_buffer.best_traj_stats(best_x=self.best_x_buffer)
            
            use_mean = False
            
            # adapt exploration exploitation:
            self.running_buffer_mean.append(mean_returns)
            # if np.std(self.running_buffer_mean) < 0.5:
            #      use_mean = True
            # if std_returns < 10:
            #      use_mean = True
            # if self.deter_iter % 10 == 0:
            #      use_mean = True
            
            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.action_range,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                stochastic_policy=self.stochastic_policy,
                use_mean=use_mean,
                exploration_noise=self.exploration_noise,
                noise_sample_inter=self.noise_sample_inter,
            )
            output = {
                "aug_traj/return": np.mean(returns),
                "aug_traj/max_return": np.max(returns),
                "aug_traj/length": np.mean(lengths),
                "buffer/size": self.replay_buffer.__len__(),
                "buffer/running_mean_std": np.std(self.running_buffer_mean),
                "aug_traj/use_mean_action": np.array([use_mean], dtype=float).item()}

            #if lengths > mean_traj_lens:
            if returns > min_return:
                self.replay_buffer.add_new_trajs(trajs)
                self.aug_trajs += trajs
                self.total_transitions_sampled += np.sum(lengths)
            else:
                self.aug_trajs += trajs
                self.total_transitions_sampled += np.sum(lengths)

        return output

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                action_range=self.action_range,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                stochastic_policy=self.stochastic_policy,
                use_mean=True,
                exploration_noise=self.exploration_noise,
                noise_sample_inter=self.noise_sample_inter,
                reward_scale=self.reward_scale,
                
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=self.writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                iteration=self.pretrain_iter,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

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

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")


        if self.variant["online_rtg_adaptation"]:
            evaluation_rtg = 1.0
            online_rtg = 1.0
        else:
            evaluation_rtg = self.variant["eval_rtg"]
            online_rtg = self.variant["online_rtg"]
        
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=evaluation_rtg, # TODO: I tink this is fixed make if flexible. thats why the eval_reward is broken
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                action_range=self.action_range,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                stochastic_policy=self.stochastic_policy,
                use_mean=True,
                exploration_noise=self.exploration_noise,
                noise_sample_inter=self.noise_sample_inter,
                reward_scale=self.reward_scale,
            )
        ]
        
        if not self.variant["expert_experience"]:
            print("\n*** Prefilling Replay Buffer! ***\n")
            for i in range(self.variant["prefill_trajectories"]):
                self.prefill_buffer(online_envs=online_envs)
                buffer_stats = self.replay_buffer.buffer_stats()

                # Update best RTG
                if buffer_stats["buffer/max_return"] > online_rtg:
                    online_rtg = buffer_stats["buffer/max_return"] * 2
                    evaluation_rtg = buffer_stats["buffer/max_return"]
                    
            print("\n*** Done Prefilling Replay Buffer! ***\n")

        while self.online_iter < self.variant["max_online_iters"] and self.total_transitions_sampled < self.variant["max_interactions"] :

            outputs = {}
            outputs["aug_traj/online_rtg"] = online_rtg
            
            self.deter_iter += 1
            # collect new trajectory 
            augment_outputs = self._augment_trajectories(
                online_envs=online_envs,
                target_explore=online_rtg,
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)

            # update online rtg if used 
            if self.variant["online_rtg_adaptation"]:
                if augment_outputs["aug_traj/max_return"] >= online_rtg:
                    online_rtg = augment_outputs["aug_traj/max_return"] * 2
                    evaluation_rtg = augment_outputs["aug_traj/max_return"]

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
                sample_policy=self.variant["sample_policy"],
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)
            outputs.update(self.replay_buffer.buffer_stats())

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns, evaluation_rtg)
                outputs.update(eval_outputs)
                outputs["evaluation/evaluation_rtg"] = evaluation_rtg
            if self.pretrain_iter + self.online_iter % 500 == 0:
                self._save_model(
                    path_prefix=self.logger.log_path,
                    iteration=self.pretrain_iter + self.online_iter,
                    is_pretrain_model=False,
                )

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=self.writer,
            )

            self.online_iter += 1


        self._save_model(
            path_prefix=self.logger.log_path,
            iteration=self.pretrain_iter + self.online_iter,
            is_pretrain_model=False,
        )

    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        # import d4rl
        if self.stochastic_policy:
            def loss_fn(
                a_hat_dist,
                a,
                attention_mask,
                entropy_reg,
                s,
                s_hat,

            ):
                # a_hat is a SquashedNormal Distribution
                log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
                entropy = a_hat_dist.entropy().mean()
                
                # state pred error
                # state_loss = ((s_hat - s)**2).mean() # [attention_mask > 0]
                # state_log_likelihood = s_hat.log_prob(s).sum(axis=2)[attention_mask > 0].mean()
                # state_entropy = s_hat.entropy().mean()
                state_entropy = 0
                state_log_likelihood = 0
                
                loss = -(log_likelihood + entropy_reg * entropy)# + state_log_likelihood + entropy_reg * state_entropy)

                return (
                    loss,
                    -log_likelihood,
                    entropy,
                    state_entropy,
                    state_log_likelihood
                )
        else:
            loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                print(f"*** Creating Environment: {env_name} ***")
                env = gym.make(env_name)# , healthy_reward=0.75)
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
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.writer = wandb.init(dir=self.logger.log_path,
                    config=self.variant,
                    project="online-dt",
                    name=self.variant["exp_name"],
                    group=self.variant["env"])


        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="HalfCheetah-v3")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    parser.add_argument("--stochastic_policy", type=int, choices=[0,1], default=0)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--use_rtg", type=int, default=0)
    parser.add_argument("--eval_rtg", type=int, default=6000)
    parser.add_argument("--num_eval_episodes", type=int, default=5)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    # Learning rate scheduler
    parser.add_argument("--use_cosineanneal", type=int, default=1, choices=[0,1])
    parser.add_argument("--warmup_steps", type=int, default=10000)    # 10000 for pure online we should probably decrease this ?!?
    # Cosine Anneal Scheduler params
    parser.add_argument("--learning_rate_reset", type=float, default=1e-2)
    parser.add_argument("--lr_restart_updates", type=int, default=100, help="Number of updating steps until resetting the learning rate")
    parser.add_argument("--decay_lr_factor", type=int, default=1, help="A factor increases Ti (number of decay steps) after a restart")
    
    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=0) # original: 1
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--expert_experience", type=int, choices=[0,1], default=0)
    parser.add_argument("--prefill_trajectories", type=int, default=50)
    parser.add_argument("--online_rtg_adaptation", type=int, choices=[0,1], default=0)
    parser.add_argument("--max_online_iters", type=int, default=30000) # 1500
    parser.add_argument("--max_interactions", type=int, default=10_000_000)
    parser.add_argument("--online_rtg", type=int, default=12000)
    parser.add_argument("--num_online_rollouts", type=int, default=1) # TODO: not used yet! always 1
    parser.add_argument("--replay_size", type=int, default=5)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=25) # 300
    parser.add_argument("--eval_interval", type=int, default=50) # 10
    
    # add to buffer when aug_return is bigger than x best mean returns
    parser.add_argument("--buffer_adding", type=str, choices=["ffo", "return", "traj_len"], default="return",
                        help="how to add new trajectories ffo=first_in_first_out, return=exchanges worst return trajectory with new traje, traj_len= exchanges shortest trajectory with new traj")
    parser.add_argument("--sample_policy", type=str, default="return", choices=["length", "return"],
                        help="Sampling strategy from the replay buffer to get training examples. Sampling based on return or trajectory length")
    parser.add_argument("--best_x", type=int, default=5)
    parser.add_argument("--exploration_noise", type=float, default=0.125)
    parser.add_argument("--noise_sample_inter", type=int, default=1,
                        help="The step frequence of how often new noise for exploration should be sampled, default=1 ")

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    
    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed) # TODO: seeding seems to be off - env seed?
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
