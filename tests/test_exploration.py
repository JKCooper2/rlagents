import unittest

from rlagents.exploration import EpsilonGreedy
from rlagents.functions.decay import FixedDecay
from gym.spaces import Discrete


class TestEpsilonGreedy(unittest.TestCase):
    def test_epsilon_property(self):
        exploration = EpsilonGreedy(Discrete(5), FixedDecay(0.2, 1, 0.1))
        self.assertEqual(0.2, exploration.value)

    def test_invalid_decay(self):
        with self.assertRaises(TypeError):
            EpsilonGreedy(Discrete(5), 1)

    def test_default_decay(self):
        exploration = EpsilonGreedy(Discrete(5))
        self.assertEqual(0.1, exploration.value)


if __name__ == '__main__':
    unittest.main()
