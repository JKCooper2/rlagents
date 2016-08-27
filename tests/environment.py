import gym

from rlagents.env_manager import EnvManager
from rlagents.pool import Pool
from rlagents.agents import Agent
from rlagents.evolution import GeneticAlgorithm
from rlagents.models import WeightedLinearModel
from rlagents.function_approximation import DefaultFA, DiscreteMaxFA, ClipFA
from rlagents.memory import ListMemory
from rlagents.exploration import DefaultExploration
from rlagents.optimisation import DefaultOptimiser


def main():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA(),
                  observation_fa=DefaultFA(),
                  memory=ListMemory(size=2, columns=['observations', 'actions', 'rewards', 'done']),
                  exploration=DefaultExploration(),
                  optimiser=DefaultOptimiser())

    pool = Pool(GeneticAlgorithm(), times_run=1)
    pool.add(agent, number=20)

    em = EnvManager("CartPole-v0", pool, api_key="sk_kH4UU0T8TgmV0K1DN8SiQ")

    em.run(n_episodes=500, video_callable=None)


if __name__ == "__main__":
    main()
