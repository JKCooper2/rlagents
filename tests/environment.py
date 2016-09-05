
from rlagents.env_manager import EnvManager
from rlagents.agent_manager import AgentManager
from rlagents.agents import Agent
from rlagents.models import WeightedLinearModel, TabularModel
from rlagents.function_approximation import DefaultFA, DiscreteMaxFA, SingleTiling
from rlagents.memory import ListMemory, PandasMemory
from rlagents.exploration import DefaultExploration, EpsilonGreedy
from rlagents.optimisation import DefaultOptimiser, TemporalDifference
from rlagents.functions.decay import FixedDecay


def main():
    agent = Agent(model=TabularModel(mean=1, std=0.05),
                  action_fa=DiscreteMaxFA(),
                  observation_fa=SingleTiling(num_tiles=6),
                  memory=ListMemory(size=2, columns=['observations', 'actions', 'rewards', 'done']),
                  exploration=EpsilonGreedy(FixedDecay(0.5, 0.995, 0.02)),
                  optimiser=TemporalDifference(learning_rate=FixedDecay(0.5, 0.995, 0.05)))

    agent = Agent(model=TabularModel(mean=1, std=0),
                  action_fa=DiscreteMaxFA(),
                  observation_fa=DefaultFA(),
                  memory=PandasMemory(size=20, columns=['observations', 'actions', 'rewards', 'done', 'new_obs']),
                  exploration=EpsilonGreedy(FixedDecay(1, 0, 1)),
                  optimiser=TemporalDifference(learning_rate=FixedDecay(0.1, 1, 0.1)))

    am = AgentManager(agent=agent)

    environments = ['BanditTenArmedUniformDistributedReward-v0',
                    'BanditTenArmedRandomFixed-v0',
                    'BanditTenArmedRandomRandom-v0',
                    'BanditTenArmedGaussian-v0',
                    'BanditTwoArmedDeterministicFixed-v0',
                    'BanditTwoArmedHighHighFixed-v0',
                    'BanditTwoArmedHighLowFixed-v0',
                    'BanditTwoArmedLowLowFixed-v0']

    for env in environments:
        am.agents[0].configured = False
        em = EnvManager(env, am)
        em.run(n_episodes=100, video_callable=None)


if __name__ == "__main__":
    main()
