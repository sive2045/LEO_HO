import argparse
import os
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from torch import nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager, A2CPolicy, ImitationPolicy
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import Actor, Critic

from gymnasium.spaces import Box

from env import LEOSATEnv

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

    def forward(
        self,
        x: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        if info is None:
            info = {}
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), state


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="a smaller gamma favors earlier win",
    )
    parser.add_argument('--debugging', type=bool, default=False)
    parser.add_argument(
        '--n-groundstations',
        type=int,
        default=10,
        help='Number of groudnstations(agents) in the env'
    )
    parser.add_argument("--n-step", type=int, default=154)
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--epoch", type=int, default=1_000_000)
    parser.add_argument("--step-per-epoch", type=int, default=1550)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--episode-per-collect", type=int, default=16)
    parser.add_argument("--repeat-per-collect", type=int, default=2)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")

    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, watch the play of pre-trained models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # a2c special
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args(), render_mode=None):
    return PettingZooEnv(LEOSATEnv(render_mode=render_mode, debugging=args.debugging))


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
    optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[BasePolicy, list[torch.optim.Optimizer], list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    if agents is None:
        agents = []
        optims = []
        for _ in range(args.n_pistons):
            # model
            net = DQN(
                observation_space.shape[2],
                observation_space.shape[1],
                observation_space.shape[0],
                device=args.device,
            ).to(args.device)

            actor = Actor(
                net,
                args.action_shape,               
                device=args.device,
            ).to(args.device)
            net2 = DQN(
                observation_space.shape[2],
                observation_space.shape[1],
                observation_space.shape[0],
                device=args.device,
            ).to(args.device)
            critic = Critic(net2, device=args.device).to(args.device)
            for m in set(actor.modules()).union(critic.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=args.lr)

            dist = torch.distributions.Categorical

            agent = A2CPolicy(
                actor=actor,
                critic=critic,
                optim=optim,
                dist_fn=dist,
                action_scaling=isinstance(env.action_space, Box),
                discount_factor=args.gamma,
                gae_lambda=args.gae_lambda,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
                reward_normalization=args.rew_norm,
                action_space=env.action_space,
            )

            agents.append(agent)
            optims.append(optim)

    policy = MultiAgentPolicyManager(agents, env, action_scaling=True, action_bound_method="clip")
    return policy, optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
    optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[dict, BasePolicy]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim, agents = get_agents(args, agents=agents, optims=optims)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,  # True
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, "LEO-HO", "A2C")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):            # 조건문 수정해야함.
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "LEO", "ac2", "policy.pth"
            )
        for i in range(args.n_groundstations):
            model_save_path = os.path.join(
                args.logdir, "LEO", "ac2", f"policy{i}.pth"
            )
            torch.save(
                policy.policies[agents[i]].state_dict(), model_save_path # 인덱스 주의
        )

    def stop_fn(mean_rewards):
        return False

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=args.resume,
    ).run()

    return result, policy


def watch(args: argparse.Namespace = get_args(), policy: BasePolicy | None = None) -> None:
    env = DummyVectorEnv([get_env])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is currently not supported",
        )
        policy, _, _ = get_agents(args)
    policy.eval()
    collector = Collector(policy, env)
    collector_result = collector.collect(n_episode=1, render=args.render)
    rews, lens = collector_result["rews"], collector_result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    result, agent = train_agent(args)
    watch(args, agent)