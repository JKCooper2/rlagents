import gym
import gym.scoreboard.scoring

from agents.tabular_q_agent import TabularQAgent
from rlagents.agents.evolutionary_agent import EvolutionaryAgent
from rlagents.models.linear import ContinuousActionLinearModel
from rlagents.models.tabular import TabularModel
from rlagents.optimisation.evolutionary.hill_climbing import HillClimbing
from rlagents.optimisation.evolutionary.genetic_algorithm import GeneticAlgorithm
from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.functions.decay import FixedDecay

ENVS = ["FrozenLake-v0",
        "FrozenLake8x8-v0",
        "Taxi-v1",
        "Roulette-v0",
        "Blackjack-v0",
        "NChain-v0",
        "CartPole-v0",
        "Acrobot-v0"]


def main():
    env = gym.make("Pendulum-v0")

    # Agent Setup
    model = TabularModel(env.action_space, env.observation_space)
    evolution = GeneticAlgorithm(scaling=2)
    batch_size = 50
    agent = EvolutionaryAgent(env.action_space, env.observation_space, model=model, evolution=evolution, batch_size=batch_size)

    out_dir = '/tmp/' + agent.name + '-results'
    env.monitor.start(out_dir, force=True, video_callable=False)

    n_episodes = 2000
    for i_episode in range(n_episodes):

        observation = env.reset()
        reward = 0
        done = False

        action = agent.act(observation, reward, done)

        while not done:
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done)

        print gym.scoreboard.scoring.score_from_local(out_dir)
        # print evolution.best_score, evolution.spread.value

    env.monitor.close()

    # gym.upload(out_dir, algorithm_id=agent.alg_id, api_key=agent.alg_id)


if __name__ == '__main__':
    main()