import unittest
import gym
import logging
import warnings

import rlagents.examples.agents


def run_agent(env, agent):
    env.monitor.start("examples/tests/agents/", force=True, video_callable=False)

    for i_episode in range(3):
        observation = env.reset()
        reward = 0
        done = False

        action = agent.act(observation, reward, done)

        while not done:
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done)


class TestAgents(unittest.TestCase):
    def test_discreteaction_continuousobservation_agents(self):
        env = gym.make("CartPole-v0")

        a = env.action_space
        o = env.observation_space

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        run_agent(env, rlagents.examples.agents.random(a, o))
        run_agent(env, rlagents.examples.agents.crossentropy_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.geneticalgorithm_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.simulatedannealing_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.hillclimbing_discretelinear(a, o))
        run_agent(env, rlagents.examples.agents.standard(a, o))
        run_agent(env, rlagents.examples.agents.standard_pandasmemory(a, o))
        # run_agent(env, rlagents.examples.agents.crossentropy_tabular(a, o))

    def test_continuousaction_continuousobservation_agents(self):
        env = gym.make("Pendulum-v0")

        a = env.action_space
        o = env.observation_space

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        run_agent(env, rlagents.examples.agents.random(a, o))
        run_agent(env, rlagents.examples.agents.crossentropy_continuouslinear(a, o))
        run_agent(env, rlagents.examples.agents.geneticalgorithm_continuouslinear(a, o))
        run_agent(env, rlagents.examples.agents.simulatedannealing_continuouslinear(a, o))
        run_agent(env, rlagents.examples.agents.hillclimbing_continuouslinear(a, o))
        # run_agent(env, rlagents.examples.agents.standard(a, o))
