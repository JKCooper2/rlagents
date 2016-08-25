import gym
from gym.scoreboard.scoring import score_from_local

"""
Env Manager is responsible for running an agent or group of agents over an environment
"""


class EnvManager(object):
    def __init__(self, env_name, agents, api_key=None):
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.agents = agents
        self.api_key = api_key

    def run(self, n_episodes=200, print_stats=True, video_callable=None, upload=False):

        self.env.monitor.start("/tmp/rlagents/", force=True, write_upon_reset=True, video_callable=video_callable)

        for i_episode in range(n_episodes):
            observation = self.env.reset()
            reward = 0
            done = False
            agent = self.agents.next_agent()

            action = agent.act(observation, reward, done)

            while not done:
                observation, reward, done, info = self.env.step(action)
                action = agent.act(observation, reward, done)

            if print_stats:
                print score_from_local("/tmp/rlagents/")

        self.env.monitor.close()

        if upload:
            gym.upload("/tmp/rlagents/", api_key=self.api_key)
