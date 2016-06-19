from rlagents import validate


class defaults:
    @staticmethod
    def exploration(action_space):
        from rlagents.exploration.epsilon_greedy import EpsilonGreedy
        from rlagents.functions.decay import FixedDecay
        return EpsilonGreedy(action_space, decay=FixedDecay(1, 0.997, 0.05))

    @staticmethod
    def learningrate():
        from rlagents.functions.decay import FixedDecay
        return FixedDecay(1, decay=0.995, minimum=0.05)

    @staticmethod
    def observationfa(observation_space):
        from gym.spaces import tuple_space, box, discrete
        from rlagents.function_approximation.discrete import Discrete
        from rlagents.function_approximation.tiles import SingleTiling

        if isinstance(observation_space, tuple_space.Tuple):
            return Discrete([space.n for space in observation_space.spaces])

        elif isinstance(observation_space, box.Box):
            return SingleTiling(observation_space, 8)

        elif isinstance(observation_space, discrete.Discrete):
            return Discrete([observation_space.n])

    @staticmethod
    def model(action_space, observation_space):
        from rlagents.models.tabular import TabularModel
        return TabularModel(action_space, observation_space)

    @staticmethod
    def history():
        from rlagents.history.history import History
        return History(size=1)


class QLearningAgent(object):
    def __init__(self, action_space, observation_space, discount=0.95, learning_rate=None, exploration=None, observation_fa=None, model=None, history=None):
        self.name = "Q Learning Agent"
        self.alg_id = "alg_OwSFZtRR2eZYkcxkG74Q"
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n

        self.discount = discount

        self.learning_rate = learning_rate if learning_rate is not None else defaults.learningrate()
        self.exploration = exploration if exploration is not None else defaults.exploration(action_space)
        self.observation_fa = observation_fa if observation_fa is not None else defaults.observationfa(observation_space)
        self.model = model if model is not None else defaults.model(action_space.n, self.observation_fa)
        self.history = history if history is not None else defaults.history()

        self.__validate_setup()

    def __validate_setup(self):
        validate.decay(self.learning_rate)
        validate.exploration(self.exploration)
        validate.model(self.model)
        validate.observation_fa(self.observation_fa)

    def __choose_action(self, observation):
        return self.exploration.choose_action(self.model, observation)

    def __learn(self, observation_key, reward, done):
        last_turn = self.history.retrieve(1)

        if last_turn is None:
            return

        prev_obs = last_turn["observations"]
        prev_action = last_turn["actions"]

        future = self.model.state_value(observation_key) if not done else 0.0
        self.model.weights[prev_obs][prev_action] += self.learning_rate.value * (reward + self.discount * future - self.model.weights[prev_obs][prev_action])

    def act(self, observation, reward, done):
        observation_key = self.observation_fa.convert(observation)

        self.__learn(observation_key, reward, done)
        action = self.__choose_action(observation_key)

        self.history.store(observation=observation_key, action=action)

        if done:
            self.exploration.update()
            self.learning_rate.update()

        return action
