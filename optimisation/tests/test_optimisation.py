import unittest

from rlagents.optimisation.evolutionary import CrossEntropy


class TestCrossEntropy(unittest.TestCase):
    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            ce = CrossEntropy(-1)

        with self.assertRaises(ValueError):
            ce = CrossEntropy(1.01)
