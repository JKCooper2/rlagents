import gym
import gym.scoreboard.scoring
from agents.dev.tabular_q_agent import TabularQAgent

ENVS = ["FrozenLake-v0",
        "FrozenLake8x8-v0",
        "Taxi-v1",
        "Roulette-v0",
        "Blackjack-v0",
        "NChain-v0",
        "CartPole-v0",
        "Acrobot-v0"]


def main():
    env = gym.make(ENVS[0])
    agent = TabularQAgent(env.action_space, env.observation_space)
    out_dir = '/tmp/' + agent.name + '-results'
    env.monitor.start(out_dir, force=True)

    n_episodes = 5000
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