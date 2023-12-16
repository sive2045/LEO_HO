"""
Train the agents.

Use Algorithm: Dueling Deep Q-Network 
Type: Multi-Agent
"""

import argparse
import os
import warnings
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

from env import LEOSATEnv

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-agent', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=10_000_000)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--debugging', type=bool, default=False)
    parser.add_argument(
        '--n-groundstations',
        type=int,
        default=10,
        help='Number of groudnstations(agents) in the env'
    )
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--target-update-freq', type=int, default=1_000)
    parser.add_argument('--epoch', type=int, default=1_000_000)
    parser.add_argument('--step-per-epoch', type=int, default=1_000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--max-reward', type=float, default=2000*155)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args(), render_mode=None):
    return PettingZooEnv(LEOSATEnv(render_mode=render_mode, debugging=args.debugging))


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    if agents is None:
        agents = []
        optims = []
        for _ in range(args.n_groundstations): 
            # model
            net = Net(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            ).to(args.device)
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            agent = DQNPolicy(
                net,
                optim,
                args.gamma,
                args.n_step,
                target_update_freq=args.target_update_freq,
                clip_loss_grad=True
            )
            agents.append(agent)
            optims.append(optim)
    
    if args.load_agent:
        for i in range(args.n_groundstations):
            agents[i].load_state_dict(torch.load(f"./log/policy{i}.pth"))
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[dict, BasePolicy]:
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
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, 'LEO', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):            # 조건문 수정해야함.
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "LEO", "dqn", "policy.pth"
            )
        for i in range(args.n_groundstations):
            model_save_path = os.path.join(
                args.logdir, "LEO", "dqn", f"policy{i}.pth"
            )
            torch.save(
                policy.policies[agents[i]].state_dict(), model_save_path # 인덱스 주의
        )

    def stop_fn(mean_rewards): 
        return mean_rewards > args.max_reward

    def train_fn(epoch, env_step):
        [agent.set_eps(args.eps_train) for agent in policy.policies.values()]

    def test_fn(epoch, env_step):
        [agent.set_eps(args.eps_test) for agent in policy.policies.values()]

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric
    )

    return result, policy


def watch(
    args: argparse.Namespace = get_args(), policy: Optional[BasePolicy] = None
) -> None:
    env = DummyVectorEnv([get_env])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is "
            "currently not supported"
        )
        policy, _, _ = get_agents(args)
    policy.eval()
    [agent.set_eps(args.eps_test) for agent in policy.policies.values()]
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=1 /30)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    result, agent = train_agent(args)
    watch(args, agent)