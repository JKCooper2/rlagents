import gym
import gym.scoreboard.scoring
import logging
import json, sys, cPickle, os
from os import path
import argparse

import numpy as np


class CrossEntropyMethod:
    def __init__(self, action_space, observation_space, batch_size=30, elite=0.3):
        self.action_space = action_space
        self.observation_space = observation_space

        self.batch_size = batch_size  # Number of samples run per batch
        self.elite = elite  # Percentage of samples selected to generate next batch
        self.elite_n = int(np.round(self.batch_size * self.elite))

        self.mean = 0.0
        self.std = 1.0

        self.batch_test = 0
        self.batch_results = []

        self.ep_reward = 0

        self.batch = self.__generate_batch()
        print self.batch

    def __generate_batch(self):
        return [np.random.normal(self.mean, self.std, len(self.observation_space.low) + 1) for _ in range(self.batch_size)]

    def __next_generation(self):
        # Select the x best scoring samples based on reverse score order
        best = np.array([self.batch[b] for b in np.argsort(self.batch_results)[-self.elite_n:]])

        # Update the mean/std based on new values
        self.mean = best.mean(axis=0)
        self.std = best.std(axis=0)

        print self.mean, self.std

        # Create a new batch
        self.batch = self.__generate_batch()

    def __choose_action(self, observation):
        y = observation.dot(self.batch[self.batch_test][:-1]) + self.batch[self.batch_test][-1]
        return int(y < 0)

    def act(self, observation, reward, done):
        action = self.__choose_action(observation)

        self.ep_reward += reward

        if done:
            self.batch_results.append(self.ep_reward)
            self.ep_reward = 0

            if len(self.batch_results) == self.batch_size:
                self.__next_generation()
                self.batch_results = []

            self.batch_test = len(self.batch_results)

        return action



class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a


def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in th_std[None, :] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        print th_std
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()

    # np.random.seed(0)
    env = gym.make('CartPole-v0')
    params = dict(n_iter=40, batch_size=20, elite_frac=0.25)
    num_steps = 200

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/cem-agent-results'
    env.monitor.start(outdir, force=True)

    # Prepare snapshotting
    # ----------------------------------------
    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
    info = {}
    info['params'] = params
    info['argv'] = sys.argv
    info['env_id'] = env.spec.id
    # ------------------------------------------

    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # Train the agent, and snapshot each stage
    for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
        print gym.scoreboard.scoring.score_from_local(outdir)
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        if args.display: do_rollout(agent, env, 200, render=True)
        writefile('agent-%.4i.pkl'%i, cPickle.dumps(agent, -1))

    # Write out the env at the end so we store the parameters of this
    # environment.
    writefile('info.json', json.dumps(info))

    env.monitor.close()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    gym.upload(outdir, algorithm_id='cem')