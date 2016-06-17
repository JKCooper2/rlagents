from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.functions.decay import FixedDecay
from rlagents.function_approximation.dev.tiles import SingleTiling
from rlagents.function_approximation.discrete import Discrete
from rlagents.models.tabular import TabularModel
from gym.spaces import discrete, tuple_space, box


class TabularQAgent(object):
    def __init__(self, action_space, observation_space, init_mean=1.0, init_std=0.2, learning_rate=None, exploration=None, discount=0.95):
        self.name = "TabularQAgent"
        self.alg_id = "alg_OwSFZtRR2eZYkcxkG74Q"
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n

        self.init_mean = init_mean
        self.init_std = init_std

        self.discount = discount

        self.learning_rate = self.__set_learning_rate(learning_rate)
        self.exploration = self.__set_exploration(action_space, exploration)

        self.step_cost = -0.01  # So agent doesn't like states it's already been in that haven't lead to a reward

        self.prev_obs = None
        self.prev_action = None

        self.fa = self.__set_fa(observation_space)

        self.model = TabularModel(action_space, observation_space)

        self.__validate_setup()

    def __validate_setup(self):
        assert hasattr(self.exploration, 'choose_action') and callable(getattr(self.exploration, 'choose_action'))
        assert hasattr(self.exploration, 'update') and callable(getattr(self.exploration, 'update'))
        assert hasattr(self.learning_rate, 'update') and callable(getattr(self.learning_rate, 'update'))
        assert hasattr(self.learning_rate, 'value')

    @staticmethod
    def __set_exploration(action_space, exploration):
        if exploration is None:
            print "Using default exploration Epsilon Greedy with decay. Start 1, decay 0.997, min 0.02"
            return EpsilonGreedy(action_space, epsilon=1, decay=0.997, minimum=0.05)

        return exploration

    @staticmethod
    def __set_learning_rate(learning_rate):
        if learning_rate is None:
            print "Using default learning rate Decay. Start 1, decay 0.995, min 0.02"
            return FixedDecay(1, decay=0.995, minimum=0.05)

        return learning_rate

    @staticmethod
    def __set_fa(observation_space):
        if isinstance(observation_space, tuple_space.Tuple):
            return Discrete([space.n for space in observation_space.spaces])

        elif isinstance(observation_space, box.Box):
            return SingleTiling(observation_space, 8)

        elif isinstance(observation_space, discrete.Discrete):
            return Discrete([observation_space.n])

    def __choose_action(self, observation):
        return self.exploration.choose_action(self.model, observation)

    def __learn(self, observation_key, reward, done):
        future = self.model.state_value(observation_key) if not done else 0.0
        self.model.weights[self.prev_obs][self.prev_action] += self.learning_rate.value * (reward + self.discount * future - self.model.weights[self.prev_obs][self.prev_action])

    def act(self, observation, reward, done):
        observation_key = self.fa.to_array(observation)

        reward += self.step_cost

        self.__learn(observation_key, reward, done)

        action = self.__choose_action(observation_key)

        self.prev_obs = observation_key
        self.prev_action = action

        if done:
            self.exploration.update()
            self.learning_rate.update()

        return action
