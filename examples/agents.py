from rlagents.agents import Agent
from rlagents.models import WeightedLinearModel, DefaultModel
from rlagents.function_approximation import DefaultFA, ClipFA, DiscreteMaxFA
from rlagents.evolution import CrossEntropy, GeneticAlgorithm, SimulatedAnnealing, HillClimbing
from rlagents.exploration import RandomExploration
from rlagents.pool import Pool


def random_discrete(action_space, observation_space):
    action_fa = DiscreteMaxFA(action_space)
    model = DefaultModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model, exploration=RandomExploration(), action_fa=action_fa)
    return agent


def random_continuous(action_space, observation_space):
    action_fa = ClipFA(action_space)
    model = DefaultModel(ClipFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model, exploration=RandomExploration(), action_fa=action_fa)
    return agent


def crossentropy_discretelinear(action_space, observation_space):
    action_fa = DiscreteMaxFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model, action_fa=action_fa)
    pool = Pool(CrossEntropy(), times_run=1)
    pool.add(agent, number=2)
    return pool


def geneticalgorithm_discretelinear(action_space, observation_space):
    action_fa = DiscreteMaxFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model, action_fa=action_fa)
    pool = Pool(GeneticAlgorithm(), times_run=1)
    pool.add(agent, number=2)
    return pool


def simulatedannealing_discretelinear(action_space, observation_space):
    action_fa = DiscreteMaxFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model, action_fa=action_fa)
    pool = Pool(SimulatedAnnealing(), times_run=1)
    pool.add(agent, number=1)
    return pool


def hillclimbing_discretelinear(action_space, observation_space):
    action_fa = DiscreteMaxFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model, action_fa=action_fa)
    pool = Pool(HillClimbing(), times_run=1)
    pool.add(agent, number=2)
    return pool


def crossentropy_continuouslinear(action_space, observation_space):
    action_fa = ClipFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=action_fa, model=model)
    pool = Pool(CrossEntropy(), times_run=1)
    pool.add(agent, number=2)
    return pool


def geneticalgorithm_continuouslinear(action_space, observation_space):
    action_fa = ClipFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=action_fa, model=model)
    pool = Pool(GeneticAlgorithm(), times_run=1)
    pool.add(agent, number=2)
    return pool


def simulatedannealing_continuouslinear(action_space, observation_space):
    action_fa = ClipFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=action_fa, model=model)
    pool = Pool(SimulatedAnnealing(), times_run=1)
    pool.add(agent, number=1)
    return pool


def hillclimbing_continuouslinear(action_space, observation_space):
    action_fa = ClipFA(action_space)
    model = WeightedLinearModel(action_fa, DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=action_fa, model=model)
    pool = Pool(HillClimbing(), times_run=1)
    pool.add(agent, number=2)
    return pool

