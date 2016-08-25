import unittest
import numpy as np

from gym.spaces import Discrete, Box, Tuple
from rlagents.function_approximation import DefaultFA, DiscreteMaxFA, ClipFA, FunctionApproximationBase


DISCRETE = Discrete(10)
BOX_1 = Box(-1.0, 1.0, (3, 4))
BOX_2 = Box(np.array([-1.0, -2.0]), np.array([2.0, 4.0]))
BOX_3 = Box(np.array([-3.0]), np.array([3]))
TUPLE = Tuple((Discrete(2), Discrete(3)))

AV_ARRAY = [-4, 5, 0, 4.8]


class TestFunctionApproximationBase(unittest.TestCase):
    def test_discrete_space(self):
        fa = FunctionApproximationBase(DISCRETE)

        self.assertEqual(fa.space_type, 'D')
        self.assertEqual(fa.num_discrete, 10)

    def test_box_space(self):
        fa = FunctionApproximationBase(BOX_1)
        self.assertEqual(fa.space_type, 'B')
        self.assertEqual(fa.num_discrete, 12)

        fa = FunctionApproximationBase(BOX_2)
        self.assertEqual(fa.space_type, 'B')
        self.assertEqual(fa.num_discrete, 2)

    def test_tuple_space(self):
        fa = FunctionApproximationBase(TUPLE)
        self.assertEqual(fa.space_type, 'T')
        self.assertEqual(fa.num_discrete, 2)


class TestDefaultFA(unittest.TestCase):
    def test_convert(self):
        fa = DefaultFA(DISCRETE)
        a = fa.convert(AV_ARRAY)

        self.assertEqual(a, AV_ARRAY)


class TestDiscreteMaxFA(unittest.TestCase):
    def test_convert(self):
        fa = DiscreteMaxFA(DISCRETE)
        a = fa.convert(AV_ARRAY)

        self.assertEqual(a, 1)


class TestClipFA(unittest.TestCase):
    def test_convert(self):
        fa = ClipFA(BOX_3)
        a = fa.convert(AV_ARRAY)

        self.assertListEqual(a.tolist(), [-3, 3, 0, 3])

        a = fa.convert([-20])
        self.assertEqual(a, [-3])