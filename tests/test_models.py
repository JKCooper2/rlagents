import unittest
import numpy as np

from gym.spaces import Discrete, Box

from rlagents.models import WeightedLinearModel
from rlagents.function_approximation import DiscreteMaxFA, DefaultFA


class TestWeightedLinearModel(unittest.TestCase):
    def setUp(self):
        self.action_fa = DiscreteMaxFA(Discrete(2))
        self.observation_fa = DefaultFA(Box(np.array([-1.0, -2.0]), np.array([2.0, 4.0])))

        self.observation = np.array([1, 2])

        np.random.seed(0)
        self.m = WeightedLinearModel(self.action_fa, self.observation_fa)

    def test_new_model(self):
        self.assertListEqual(self.m.weights.tolist(), np.array([[1.764052345967664, 0.4001572083672233], [0.9787379841057392, 2.240893199201458]]).tolist())
        self.assertListEqual(self.m.bias_weight.tolist(), np.array([[1.8675579901499675, -0.977277879876411]]).tolist())

    def test_action_value(self):
        self.assertListEqual(self.m.action_value(self.observation).tolist(), np.array([5.5890863043291095, 3.904665726893728]).tolist())

    def test_state_value(self):
        self.assertEqual(self.m.state_value(self.observation), 5.5890863043291095)

    def test_state_action_value(self):
        with self.assertRaises(TypeError):
            self.m.state_action_value(self.observation, 1.5)

        self.assertEqual(self.m.state_action_value(self.observation, 0), 5.5890863043291095)
        self.assertEqual(self.m.state_action_value(self.observation, 1), 3.904665726893728)

    def test_export_values(self):
        self.assertListEqual(self.m.export_values().tolist(), np.array([1.764052345967664, 0.4001572083672233, 0.9787379841057392, 2.240893199201458, 1.8675579901499675, -0.977277879876411]).tolist())
        self.assertFalse(self.m.export_values() is np.concatenate((self.m.weights, self.m.bias_weight)))

    def test_import_values(self):
        with self.assertRaises(ValueError):
            self.m.import_values(np.array([1, 2, 3, 4]))

        new_weights = np.array([1, 2, 3, 4, 5, 6])

        self.m.import_values(new_weights)

        self.assertEqual(self.m.weights.tolist(), np.array([[1, 2], [3, 4]]).tolist())
        self.assertEqual(self.m.bias_weight.tolist(), np.array([[5, 6]]).tolist())

    def test_reset(self):
        self.assertEqual(self.m.weights.tolist(), np.array([[1.764052345967664, 0.4001572083672233], [0.9787379841057392, 2.240893199201458]]).tolist())
        self.m.reset()
        self.assertNotEqual(self.m.weights.tolist(), np.array([[1.764052345967664, 0.4001572083672233], [0.9787379841057392, 2.240893199201458]]).tolist())

    def test_bias_false(self):
        self.m = WeightedLinearModel(self.action_fa, self.observation_fa, bias=False)
        self.assertEqual(self.m.bias_weight.tolist(), np.array([[0.0, 0.0]]).tolist())

    def test_bias_false_action_value(self):
        self.m = WeightedLinearModel(self.action_fa, self.observation_fa, bias=False)
        self.assertEqual(self.m.action_value(self.observation).tolist(), np.array([0.7436507139384737, 0.6698397955790467]).tolist())
