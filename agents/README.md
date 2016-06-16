Agents are completed systems capable of taking in information from the environment and returning actions
They are essentially a shell in which behaviours/tendencies can be placed inside

Agents can be initialised using `agent(action_space, observation_space)`
Other initialisation variables will be optional parameters

Agents must implement:
act(observation, reward, done)