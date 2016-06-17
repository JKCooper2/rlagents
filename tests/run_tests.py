import unittest

from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.functions.decay import FixedDecay
from gym.spaces import Discrete


class TestEpsilonGreedy(unittest.TestCase):
    def test_epsilon_property(self):
        exploration = EpsilonGreedy(Discrete(5), FixedDecay(0.1, 1, 0.1))
        self.assertEqual(0.1, exploration.epsilon)

    def test_invalid_action_space(self):
        with self.assertRaises(NotImplementedError):
            exploration = EpsilonGreedy(1, FixedDecay(0.1, 1, 0.1))

    def test_invalid_decay(self):
        with self.assertRaises(NotImplementedError):
            exploration = EpsilonGreedy(Discrete(5), 1)

    def test_default_decay(self):
        exploration = EpsilonGreedy(Discrete(5))
        self.assertEqual(0.1, exploration.epsilon)
        self.assertTrue(callable(getattr(exploration._decay, "update", None)))

    def test_invalid_model(self):
        exploration = EpsilonGreedy(Discrete(5))
        with self.assertRaises(NotImplementedError):
            exploration.choose_action(5, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
