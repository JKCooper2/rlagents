# Title: Multi Armed Bandits and Exploration Strategies
# Author: Sudeep Raja
# URL: http://sudeepraja.github.io/Bandits/
# UCB and Thompson Sampling are not yet in rlagents

import matplotlib.pyplot as plt
import copy

from gym.monitoring.monitor import load_results

from rlagents.env_manager import EnvManager
from rlagents.agent_manager import AgentManager
from rlagents.agents import Agent
from rlagents.models import TabularModel
from rlagents.function_approximation import DiscreteMaxFA
from rlagents.exploration import EpsilonGreedy, RandomExploration, Softmax
from rlagents.optimisation import TemporalDifference
from rlagents.functions.decay import FixedDecay


def main():
    # All agents use  a tabular model will initial values of 0
    # Updates are done via TD learning with a fixed learning rate
    # Action_FA of discrete max means the agent chooses the action with the highest utility from a discrete array
    base_agent = Agent(model=TabularModel(mean=0, std=0),
                       action_fa=DiscreteMaxFA(),
                       optimiser=TemporalDifference(learning_rate=FixedDecay(0.2)))

    # Randomly select the next action
    random_agent = copy.deepcopy(base_agent)
    random_agent.exploration = RandomExploration()

    # Always select the best action seen so far (is default behaviour for agents)
    greedy_agent = copy.deepcopy(base_agent)

    # Always select the best action seen so far with optimistic starting values
    optimistic_greedy_agent = copy.deepcopy(base_agent)
    optimistic_greedy_agent.model = TabularModel(mean=1, std=0)

    # Select a random action with decaying likelihood
    egreedy_agent = copy.deepcopy(base_agent)
    egreedy_agent.exploration = EpsilonGreedy(FixedDecay(1, 0.995, 0.01))

    # Select a random action with fixed likelihood
    fixed_egreedy_agent = copy.deepcopy(base_agent)
    fixed_egreedy_agent.exploration = EpsilonGreedy(FixedDecay(0.2))

    # Explores using softmax
    boltzmann_agent = copy.deepcopy(base_agent)
    boltzmann_agent.exploration = Softmax(FixedDecay(2, 0.995, 0.1))

    agents = [random_agent, greedy_agent, optimistic_greedy_agent, egreedy_agent, fixed_egreedy_agent, boltzmann_agent]
    labels = ['Random', 'Greedy', 'Optimistic Greedy', 'E-Greedy Decay', 'E-Greedy Fixed', 'Boltzmann']

    agent_reward = []
    max_reward = []
    episodes = 100

    for agent in agents:
        path = "/tmp/rlagents/"
        am = AgentManager(agent=agent)
        em = EnvManager('BanditTenArmedUniformDistributedReward-v0', am)
        em.run(n_episodes=episodes, print_stats=False, path=path, video_callable=False)

        max_reward.append(max(em.env.r_dist))
        results = load_results(path)
        agent_reward.append(results['episode_rewards'])

    for i, ar in enumerate(agent_reward):
        percent_correct = [agent_reward[i][:j].count(max_reward[i])/float(j) for j in range(1, episodes)]
        plt.plot(range(1, episodes), percent_correct, label=labels[i])

    plt.xlabel('Steps')
    plt.ylabel('% Optimal Arm Pulls')
    plt.ylim(-0.2, 1.5)
    plt.legend(loc=2)

    plt.show()


if __name__ == "__main__":
    main()
