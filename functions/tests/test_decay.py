import unittest

from rlagents.functions.decay import DecayBase, FixedDecay, EpisodeNumber


class TestDecayBase(unittest.TestCase):
    def test_valid_init(self):
        decay = DecayBase(1, 0.2, 0)
        self.assertEqual(decay.value, 1)
        self.assertEqual(decay.decay, 0.2)
        self.assertEqual(decay.minimum, 0)

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            decay = DecayBase(-1, 0.2, 0)


if __name__ == '__main__':
    unittest.main()
