# rl-agents
Reinforcement Learning Agents



Agents are completed systems capable of taking in information from the environment and returning actions
They are essentially a shell in which behaviours/tendencies can be placed inside

Agents can be initialised using `agent(action_space, observation_space)`
Other initialisation variables will be optional parameters

Agents must implement:
act(observation, reward, done)




Exploration looks after choosing suboptimal actions

All exploration functions must implement:
- choose_action(representation, observation)
- update()



Function approximation refers to methods for converting continuous spaces into a useable format



Models are the representations of knowledge

Must implement:
 - score(observation) -> Returns a single value
 - export_values() -> Convert representation to array
 - import_values(array) -> Convert array to representation
 - reset() -> Recreates the model independent of previous states