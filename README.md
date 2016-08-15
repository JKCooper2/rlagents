# Reinforcement Learning Agents
RL Agents is a library created in order to break apart reinforcement learning
agents into their core components, and allow for each of these components to
be plugged in and out as needed.

NOTE: At the moment still very much a pet project WIP, all implementation details can change at any point


## Structure
The goal of the structure is to be as modular as possible, that every class
that makes up an agent can be replaced by an object with a similar Base.
This will likely come at the expense of speed, but will make development,
comparison, and improvement of agents much faster


### Agent
The agent is the base object that everything is housed inside.

There are two types of agents included:

1. Evolutionary Agent: Optimise their models by combining/mutating the
models values based on the total rewards from a pool of agents
2. Exploratory Agent: Optimise their models through an exploration bias
that alters there behaviour in a particular state from the current beliefs. The
model values are then updated based on the new states seen

Evolutionary agents are sometimes not considered as true RL agents, but they
are included here as they are able to complete RL tasks.


### Memory
Stores state history in a way that allows agents to add or remove items and classes of items as needed.
Example items that would be stored:

  * Observations
  * Actions
  * Rewards
  * Done (End of episodes)


### Model
A model is the structure and values of the agents knowledge. References the agents memory
and uses that to return actions/action-values


### Exploration
Exploration is a bias function that alters the agents current action-value
function which results in the agent choosing what it believes to be sub-optimal
actions. This is required so agents don't get stuck in local minima


### Function Approximation
Function approximation refers to methods for converting spaces
into a useable format, e.g. Tiling a continuous space to allow an agent
that uses a discrete style model like a table to work with the environment


### Optimisation
Optimisation is process of updating a model. Evolutionary methods only use
the current pool of agents' model values and rewards. Exploratory agents will use
current agents model combined will any values stored in memory


### Functions
Functions contain support functions commonly used by various agents. Includes things like:

  * Decay Functions