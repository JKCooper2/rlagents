
from rlagents.env_manager import EnvManager
from rlagents.agent_manager import AgentManager
from rlagents.agents import Agent
from rlagents.models import WeightedLinearModel, TabularModel
from rlagents.function_approximation import DefaultFA, DiscreteMaxFA, SingleTiling
from rlagents.memory import ListMemory
from rlagents.exploration import DefaultExploration, EpsilonGreedy
from rlagents.optimisation import DefaultOptimiser, TemporalDifference
from rlagents.functions.decay import FixedDecay


def main():
    agent = Agent(model=TabularModel(mean=1, std=0.05),
                  action_fa=DiscreteMaxFA(),
                  observation_fa=SingleTiling(num_tiles=6),
                  memory=ListMemory(size=2, columns=['observations', 'actions', 'rewards', 'done']),
                  exploration=EpsilonGreedy(FixedDecay(0.5, 0.995, 0.02)),
                  optimiser=MonteCarlo(learning_rate=FixedDecay(0.5, 0.995, 0.05)))

    agent = Agent(model=TabularModel(mean=0, std=0.05),
                  action_fa=DiscreteMaxFA(),
                  observation_fa=DefaultFA(),
                  memory=ListMemory(size=2, columns=['observations', 'actions', 'rewards', 'done']),
                  exploration=EpsilonGreedy(FixedDecay(0.8, 0.997, 0.05)),
                  optimiser=TemporalDifference(learning_rate=FixedDecay(1, 0.995, 0.1)))

    am = AgentManager(agent=agent)

    em = EnvManager("FrozenLake-v0", am)

    em.run(n_episodes=2000, video_callable=None)

if __name__ == "__main__":
    main()
