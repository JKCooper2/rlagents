import warnings
import copy

from rlagents.evolution import EvolutionaryBase, HillClimbing

"""
Pool holds a collection of agents and allows for aggregate functions to performed over that collection, e.g. evolution
"""


class Pool(object):
    def __init__(self, evolution, times_run=1):
        self.evolution = evolution
        self.times_run = times_run

        self.agents = []
        self.agent_results = []

    @property
    def batch_size(self):
        return len(self.agents) / self.times_run

    @property
    def agents_tested(self):
        return len(self.agent_results)

    @property
    def evolution(self):
        return self._evolution

    @evolution.setter
    def evolution(self, e):
        if not isinstance(e, EvolutionaryBase):
            e = HillClimbing()
            warnings.warn("Evolution not subclass of EvolutionaryBase, using default HillClimbing")

        self._evolution = e

    @property
    def times_run(self):
        return self._times_run

    @times_run.setter
    def times_run(self, tr):
        if not isinstance(tr, int) or tr < 1:
            raise ValueError("Times must be an int >= 1")

        self._times_run = tr

    def add(self, agent, number=1):
        for i in range(number):
            new_agent = copy.deepcopy(agent)

            for j in range(self.times_run):
                self.agents.append(copy.deepcopy(new_agent))

    def clear(self):
        self.agents = []

    def next_agent(self):
        """Called by EnvManager to get the next agent to run"""
        # Check if the agent has any record
        current_agent = self.agents_tested % self.batch_size

        if self.agents[current_agent].done:
            self.agent_results.append(self.agents[current_agent].episode_reward)

        # If not all agents tested then return the next agent
        if self.agents_tested < self.batch_size * self.times_run:
            return self.agents[self.agents_tested % self.batch_size]

        # Otherwise generate the next batch
        next_batch = self.evolution.next_generation(*self.__compress_batch())

        for i in range(self.batch_size):
            for j in range(self.times_run):
                ind = i * self.times_run + j
                self.agents[ind].model = next_batch[i]
                self.agents[ind].done = False
                self.agents[ind].episode_reward = 0

        self.agent_results = []
        return self.agents[0]

    def __compress_batch(self):
        """Combines batches and results together when repeats are used"""
        models = [agent.model for agent in self.agents]

        # If no repeats then compress is unnecessary
        if self.times_run == 1:
            return models, self.agent_results

        batch = []
        batch_results = []

        for i in range(self.batch_size):
            index = i * self.times_run
            batch.append(models[index])
            batch_results.append(float(sum(self.agent_results[index:index + self.times_run])) / self.times_run)

        return batch, batch_results
