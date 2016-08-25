import gym
import gym_pull
import gym_ple
from gym.scoreboard.scoring import score_from_local

import rlagents.examples.agents


def main():
    env = gym.make("CartPole-v0")

    # agent = rlagents.examples.agents.crossentropy_discretelinear(env.action_space, env.observation_space)
    # agent.times_run = 1
    # agent.batch_size = 20
    #
    # agent = rlagents.examples.agents.standard(env.action_space, env.observation_space)
    # agent = rlagents.examples.agents.random(env.action_space, env.observation_space)
    # agent = rlagents.examples.agents.geneticalgorithm_continuouslinear(env.action_space, env.observation_space)
    # agent = rlagents.examples.agents.geneticalgorithm_discretelinear(env.action_space, env.observation_space)
    # agent.batch_size = 20
    # agent.times_run = 1

    agent = rlagents.examples.agents.random(env.action_space, env.observation_space)

    env.monitor.start("examples/tests/agents/", force=True, write_upon_reset=True, video_callable=False)

    for i_episode in range(2000):
        observation = env.reset()
        reward = 0
        done = False

        action = agent.act(observation, reward, done)

        while not done:
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done)

        print score_from_local("examples/tests/agents/")

    env.monitor.close()
    # gym.upload("examples/tests/agents/", api_key="sk_kH4UU0T8TgmV0K1DN8SiQ")


if __name__ == "__main__":
    main()
