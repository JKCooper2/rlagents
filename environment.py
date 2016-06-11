import gym
import gym.scoreboard.scoring
from agents.dev.ContinuousTimeRBF import RBFAgent


def main():
    env = gym.make('CartPole-v0')
    agent = RBFAgent(env.action_space, env.observation_space)
    out_dir = '/tmp/rbf-results'
    env.monitor.start(out_dir, force=True)

    n_episodes = 2000
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


if __name__ == '__main__':
    main()