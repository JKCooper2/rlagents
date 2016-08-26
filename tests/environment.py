import gym

from rlagents.env_manager import EnvManager
from rlagents.pool import Pool
from rlagents.agents import Agent
from rlagents.evolution import GeneticAlgorithm
from rlagents.models import WeightedLinearModel
from rlagents.function_approximation import DefaultFA, DiscreteMaxFA
import rlagents.examples.agents

def main():
    env = gym.make("Pendulum-v0")

    action_fa = DiscreteMaxFA(env.action_space)
    observation_fa = DefaultFA(env.observation_space)

    agent = Agent(env.action_space, env.observation_space, model=WeightedLinearModel(action_fa, observation_fa))
    pool = Pool(GeneticAlgorithm(), times_run=1)
    pool.add(agent, number=20)

    agent = rlagents.examples.agents.random_continuous(env.action_space, env.observation_space)

    em = EnvManager("Pendulum-v0", agent, api_key="sk_kH4UU0T8TgmV0K1DN8SiQ")

    em.run(n_episodes=500, video_callable=None)


if __name__ == "__main__":
    main()
