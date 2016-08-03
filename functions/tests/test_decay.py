import unittest

from rlagents.functions.decay import DecayBase, FixedDecay, EpisodicDecay


class TestDecayBase(unittest.TestCase):
    def test_valid_init(self):
        decay = DecayBase(1, 0.2, 0)
        self.assertEqual(decay.value, 1)
        self.assertEqual(decay.decay, 0.2)
        self.assertEqual(decay.minimum, 0)
        self.assertTrue(decay.can_update)

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            DecayBase(-1, 0.2, 0)
        with self.assertRaises(ValueError):
            DecayBase(1, -0.2, 0)
        with self.assertRaises(ValueError):
            DecayBase(1, 0.2, -0.5)

    def test_stop(self):
        decay = DecayBase(1, 0.2, 0)
        decay.stop()
        self.assertFalse(decay.can_update)


class TestFixedDecay(unittest.TestCase):
    def test_update(self):
        decay = FixedDecay(1, 0.95, 0)
        self.assertEqual(decay.value, 1)
        decay.update()
        self.assertEqual(decay.value, 0.95)
        decay.update()
        self.assertEqual(decay.value, 0.9025)


class TestEpisodicDecay(unittest.TestCase):
    def test_update(self):
        decay = EpisodicDecay(0.99, 0)
        self.assertEqual(decay.value, 1)
        self.assertEqual(decay.episode_number, 0)

        decay.update()
        self.assertEqual(decay.value, 1)
        self.assertEqual(decay.episode_number, 1)

        decay.update()
        self.assertEqual(round(decay.value, 4), 0.5035)
        self.assertEqual(decay.episode_number, 2)

        decay.update()
        self.assertEqual(round(decay.value, 4), 0.3370)
        self.assertEqual(decay.episode_number, 3)

