# Reinforcement Learning Agents
RL Agents is a library created in order to break apart reinforcement learning
agents into there core components, and allow for each of these components to
be plugged in and out as needed.


## Structure

### Agent
The agent is the base object that everything is housed inside.
There are two types of agents:
1. Evolutionary Agent - Optimises their models by combining/mutating the
models values based on the total rewards from a pool of agents
2. Exploratory Agent - Optimises their models through an exploration bias
that alters there behaviour in a particular state from the base beliefs. The
model values are then updated based on the new states seen

Evolutionary agents are sometimes not considered as true RL agents, but they
are included here as they are able to complete RL tasks.

Interface:
```python
act(observation, reward, done)
```


### History
History stores the memories of each agent


### Model
A model is the structure and values of the agents knowledge. An observation
is fed into the model and it outputs the agent valuation of the current state/actions

Interface:
```python
score(observation)
export_values()
import_values()
reset()
```


### Exploration
Exploration is a bias function that alters the agents current action-value
function which results in the agent choosing what it believes to be sub-optimal
actions.

Interface:
```python
update()
choose_action(model, observation)
```


### Function Approximation
Function approximation refers to methods for converting spaces
into a useable format, e.g. Tiling a continuous space to allow an agent
that uses a discrete style model like a table to work with the environment

Interface:
```python
num_discrete()
convert(observation)
```


### Optimisation
Optimisation is process of updating a model


### Functions
Functions contain support functions commonly used by various agents
This includes:
- Decay Functions