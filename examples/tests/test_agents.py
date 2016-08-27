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
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        env = "CartPole-v0"
        run_agent(env, rlagents.examples.agents.random_discrete())
        run_agent(env, rlagents.examples.agents.crossentropy_discretelinear())
        run_agent(env, rlagents.examples.agents.geneticalgorithm_discretelinear())
        run_agent(env, rlagents.examples.agents.simulatedannealing_discretelinear())
        run_agent(env, rlagents.examples.agents.hillclimbing_discretelinear())

    def test_continuousaction_continuousobservation_agents(self):
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        env = "Pendulum-v0"
        run_agent(env, rlagents.examples.agents.random_continuous())
        run_agent(env, rlagents.examples.agents.crossentropy_continuouslinear())
        run_agent(env, rlagents.examples.agents.geneticalgorithm_continuouslinear())
        run_agent(env, rlagents.examples.agents.simulatedannealing_continuouslinear())
        run_agent(env, rlagents.examples.agents.hillclimbing_continuouslinear())