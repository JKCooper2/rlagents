
from rlagents.env_manager import EnvManager
from rlagents.agent_manager import AgentManager
from rlagents.agents import Agent
from rlagents.models import WeightedLinearModel, TabularModel
from rlagents.function_approximation import DefaultFA, DiscreteMaxFA, SingleTiling
from rlagents.memory import ListMemory, PandasMemory
from rlagents.exploration import DefaultExploration, EpsilonGreedy
from rlagents.optimisation import DefaultOptimiser, TemporalDifference, MonteCarlo
from rlagents.functions.decay import FixedDecay


def main():
    agent = Agent(model=TabularModel(mean=1, std=0.00),
                  action_fa=DiscreteMaxFA(),
                  observation_fa=SingleTiling(num_tiles=8),
                  memory=PandasMemory(size=1, columns=['observations', 'actions', 'rewards', 'done', 'new_obs']),
                  exploration=EpsilonGreedy(FixedDecay(0.5, 0.99, 0.05)),
                  optimiser=MonteCarlo(learning_rate=FixedDecay(0.2, 1, 0.02)))

    # agent = Agent(model=TabularModel(mean=1, std=0),
    #               action_fa=DiscreteMaxFA(),
    #               observation_fa=DefaultFA(),
    #               memory=PandasMemory(size=20, columns=['observations', 'actions', 'rewards', 'done', 'new_obs']),
    #               exploration=EpsilonGreedy(FixedDecay(1, 0, 1)),
    #               optimiser=TemporalDifference(learning_rate=FixedDecay(0.1, 1, 0.1)))

    am = AgentManager(agent=agent)

    em = EnvManager('CartPole-v0', am)
    em.run(n_episodes=500, video_callable=None)


if __name__ == "__main__":
    main()
