"""
Enviornment Test

"""
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from env import LEOSATEnv


if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment (selecte rendoer_mode)
    env = LEOSATEnv(render_mode="human", debugging=True)

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent -> Random policies
    agents = [RandomPolicy()] * 10
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