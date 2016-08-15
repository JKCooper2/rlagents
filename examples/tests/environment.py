import gym
from gym.scoreboard.scoring import score_from_local

import rlagents.examples.agents


def main():
    env = gym.make("CartPole-v0")

    agent = rlagents.examples.agents.crossentropy_discretelinear(env.action_space, env.observation_space)
    agent.times_run = 2
    agent.batch_size = 20

    env.monitor.start("examples/tests/agents/", force=True, video_callable=False)

    for i_episode in range(500):
        observation = env.reset()
        reward = 0
        done = False

        action = agent.act(observation, reward, done)

        while not done:
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done)

        print score_from_local("examples/tests/agents/")

if __name__ == "__main__":
    main()
