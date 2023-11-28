"""
Enviornment Test

TODO
1. 테스트 환경 정리하기 --> args 형태로
2. history 및 결과 그래프 산출
"""
import torch
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, DQNPolicy
from tianshou.utils.net.common import Net
import gymnasium as gym

from env import LEOSATEnv


if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment (selecte rendoer_mode)
    #env = LEOSATEnv(render_mode="human", debugging=True, interference_mode=True)
    env = LEOSATEnv(debugging=True, interference_mode=True)

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent -> Random policies
    #agents = [RandomPolicy()] * 10
    agents = []
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    state_shape = observation_space.shape or observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    for _ in range(10): 
        # model
        net = Net(
            state_shape,
            action_shape,
            hidden_sizes=[128, 128, 256],
            device='cuda'
        ).to('cuda')
        optim = torch.optim.Adam(net.parameters(), lr=0.001)
        agent = DQNPolicy(
            net,
            optim,
            0.9,
            10,
            target_update_freq=1000
        )
        agents.append(agent)

    for i in range(10):
        agents[i].load_state_dict(torch.load(f"./log/policy{i}.pth"))
    policies = MultiAgentPolicyManager(agents, env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode
    result = collector.collect(n_episode=1)

    # Print results
    print(
        f"# of total episodes: {result['n/ep']}\n",
        f"step count         : {result['n/st']}\n",
        f"rewards            : {result['rews'].flatten()}\n",
        f"rweards mean       : {result['rew']}\n",
    )