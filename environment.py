import gym
import gym.scoreboard.scoring

import rlagents.examples.agents

ENVS = ["CartPole-v0",
        "Acrobot-v0",
        "OffSwitchCartpole-v0",
        "SemisuperPendulumNoise-v0",
        "SemisuperPendulumRandom-v0",
        "SemisuperPendulumDecay-v0"]


def main():
    env = gym.make(ENVS[0])

    agent = rlagents.examples.agents.hillclimbing_discretelinear(env.action_space, env.observation_space)

    out_dir = '/tmp/' + agent.name + '-results'
    env.monitor.start(out_dir, force=True)

    n_episodes = 1000
    for i_episode in range(n_episodes):

        observation = env.reset()
        reward = 0
        done = False

        action = agent.act(observation, reward, done)

        while not done:
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done)

        print gym.scoreboard.scoring.score_from_local(out_dir)

    env.monitor.close()

    # gym.upload(out_dir, algorithm_id=agent.alg_id, api_key="sk_kH4UU0T8TgmV0K1DN8SiQ")


if __name__ == '__main__':
    main()
