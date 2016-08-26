import unittest
import gym
import logging
import warnings

import rlagents.examples.agents
from rlagents.env_manager import EnvManager


def run_agent(env, agents):
    em = EnvManager(env, agents)
    em.run(n_episodes=3, video_callable=False, print_stats=False)


class TestAgents(unittest.TestCase):
    def test_discreteaction_continuousobservation_agents(self):
        env = "CartPole-v0"
        genv = gym.make(env)

        a = genv.action_space
        o = genv.observation_space

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        run_agent(env, rlagents.examples.agents.random_discrete(a, o))
        run_agent(env, rlagents.examples.agents.crossentropy_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.geneticalgorithm_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.simulatedannealing_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.hillclimbing_discretelinear(a, o))

    def test_continuousaction_continuousobservation_agents(self):
        env = "Pendulum-v0"
        genv = gym.make(env)

        a = genv.action_space
        o = genv.observation_space

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        run_agent(env, rlagents.examples.agents.random_continuous(a, o))
        run_agent(env, rlagents.examples.agents.crossentropy_continuouslinear(a, o))
        run_agent(env, rlagents.examples.agents.geneticalgorithm_continuouslinear(a, o))
        run_agent(env, rlagents.examples.agents.simulatedannealing_continuouslinear(a, o))
        run_agent(env, rlagents.examples.agents.hillclimbing_continuouslinear(a, o))