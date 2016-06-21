import numpy as np
from gym.spaces import discrete


class EpisodicAgent:
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        assert isinstance(action_space, discrete.Discrete), 'unsupported action space for now.'

        self.name = "KNN"

        # options
        self.epsilon = 1.0  # probability of choosing a random action
        self.epsilon_decay = 0.98  # decay of epsilon per episode
        self.epsilon_min = 0
        self.nnfind = 500  # how many nearest neighbors to consider in the policy?
        self.mem_needed = 500  # amount of data to have before we can start exploiting
        self.mem_size = 50000  # maximum size of memory
        self.gamma = 0.95  # discount factor

        # internal vars
        self.iter = 0
        self.mem_pointer = 0  # memory pointer
        self.max_pointer = 0
        self.db = None  # large array of states seen
        self.dba = {}  # actions taken
        self.dbr = {}  # rewards obtained at all steps
        self.dbv = {}  # value function at all steps, computed retrospectively
        self.ep_start_pointer = 0

    def act(self, observation, reward, done):
        assert isinstance(observation, np.ndarray) and observation.ndim == 1, 'unsupported observation type for now.'

        if self.db is None:
            # lazy initialization of memory
            self.db = np.zeros((self.mem_size, observation.size))
            self.mem_pointer = 0
            self.ep_start_pointer = 0

        # we have enough data, we want to explore, and we have seen at least one episode already (so values were computed)
        if self.iter > self.mem_needed and np.random.rand() > self.epsilon and self.dbv:
            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            ds = np.sum((self.db[:self.max_pointer] - observation)**2, axis=1) # L2 distance
            ix = np.argsort(ds) # sorts ascending by distance
            ix = ix[:min(len(ix), self.nnfind)] # crop to only some number of nearest neighbors

            # find the action that leads to most success. do a vote among actions
            adict = {}
            ndict = {}
            for i in ix:
                vv = self.dbv[i]
                aa = self.dba[i]
                vnew = adict.get(aa, 0) + vv
                adict[aa] = vnew
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.iteritems()]
            its.sort(reverse=True)  # descending
            a = its[0][1]

        else:
            # explore: do something random
            a = self.action_space.sample()

        # record move to database
        if self.mem_pointer < self.mem_size:
            self.db[self.mem_pointer] = observation  # save the state
            self.dba[self.mem_pointer] = a  # and the action we took
            self.dbr[self.mem_pointer-1] = reward  # and the reward we obtained last time step
            self.dbv[self.mem_pointer-1] = 0
        self.mem_pointer += 1
        self.iter += 1

        if done:  # episode Ended;

            # compute the estimate of the value function based on this rollout
            v = 0
            for t in reversed(xrange(self.ep_start_pointer, self.mem_pointer)):
                v = self.gamma * v + self.dbr.get(t,0)
                self.dbv[t] = v

            self.ep_start_pointer = self.mem_pointer
            self.max_pointer = min(max(self.max_pointer, self.mem_pointer), self.mem_size)

            # decay exploration probability
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)  # cap at epsilon_min

        return a
