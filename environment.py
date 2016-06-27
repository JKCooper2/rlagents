import gym
import gym.scoreboard.scoring

from rlagents.agents.dev.knn.KNN2 import EpisodicAgent


def main():
    env = gym.make("CartPole-v0")
    out_dir = '/tmp/open-ai-results'
    env.monitor.start(out_dir, force=True)

    # Agent Initialisation
    agent = EpisodicAgent(env.action_space, env.observation_space)

    for episode in range(1000):

        observation = env.reset()
        reward = 0
        done = False

        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        # Receive the last observation/reward/done state
        agent.act(observation, reward, done)

        print gym.scoreboard.scoring.score_from_local(out_dir)

    env.monitor.close()
    # gym.upload(out_dir, algorithm_id=agent.alg_id, api_key="sk_kH4UU0T8TgmV0K1DN8SiQ")


if __name__ == '__main__':
    main()
