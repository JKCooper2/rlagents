import unittest

from rlagents.exploration import EpsilonGreedy, Softmax
from rlagents.functions.decay import FixedDecay, DecayBase
from gym.spaces import Discrete


class TestEpsilonGreedy(unittest.TestCase):
    def test_default_decay(self):
        exploration = EpsilonGreedy(1)
        self.assertTrue(isinstance(exploration.decay, DecayBase))

    def test_epsilon_property(self):
        exploration = EpsilonGreedy(FixedDecay(0.2, 0.95, 0.1))
        self.assertEqual(0.2, exploration.value)

    def test_update(self):
        exploration = EpsilonGreedy(FixedDecay(1, 0.95, 0.1))
        exploration.update()

        self.assertEqual(exploration.value, 0.95)


class TestSoftmax(unittest.TestCase):
    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            Softmax(-1)
        with self.assertRaises(ValueError):
            Softmax(2)
