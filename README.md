# Reinforcement Learning Agents
RL Agents is a library created in order to break apart reinforcement learning
agents into their core components, and allow for each of these components to
be plugged in and out as needed.

NOTE: At the moment still very much a pet project WIP, all implementation details
can change at any point


## Structure
The goal of the structure is to be as modular as possible, that every class
that makes up an agent can be replaced by an object with a similar Base.
This will likely come at the expense of speed, but will make development,
comparison, and improvement of agents much faster


### Agent
An agent is a self-contained being in the world. It is the shell object to
which everything is based around


### Memory
Stores state history in a way that allows agents to add or remove items and
classes of items as needed.


### Model
A model is the structure and values of the agents knowledge. References the
agents memory and uses that to return actions/action-values


### Exploration
Exploration is a bias function that alters the agents current action-value
function.


### Function Approximation
Function approximation refers to methods for converting spaces
into a usable format


### Optimisation
Optimisation is process of updating an individual model.


### Evolution
Evolution updates a pool of agents using aggregated information about
models and performance


### Pool
Pool holds multiple agents and allows for easy interactions between them


### Env Manager
Env Manager is responsible for running an agent or group of agents over
an environment or set of environments


### Functions
Functions contain commonly used support functions and classes