import gym
import gym.scoreboard.scoring

from agents.tabular_q_agent import TabularQAgent
from agents.evolutionary_agent import EvolutionaryAgent

ENVS = ["FrozenLake-v0",
        "FrozenLake8x8-v0",
        "Taxi-v1",
        "Roulette-v0",
        "Blackjack-v0",
        "NChain-v0",
        "CartPole-v0",
        "Acrobot-v0"]


def main():
    env = gym.make("CartPole-v0")
    agent = EvolutionaryAgent(env.action_space, env.observation_space)
    out_dir = '/tmp/' + agent.name + '-results'
    env.monitor.start(out_dir, force=True, video_callable=False)

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

    # gym.upload(out_dir, algorithm_id=agent.alg_id, api_key=agent.alg_id)


if __name__ == '__main__':
    main()