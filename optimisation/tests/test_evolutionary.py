import unittest
import numpy as np

from rlagents.optimisation.evolutionary import CrossEntropy, GeneticAlgorithm, HillClimbing, SimulatedAnnealing
from rlagents.functions.decay import FixedDecay

class TestCrossEntropy(unittest.TestCase):
    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            ce = CrossEntropy(-1)

        with self.assertRaises(ValueError):
            ce = CrossEntropy(1.01)


class TestGeneticAlgorithm(unittest.TestCase):
    def test_valid_init(self):
        GeneticAlgorithm()

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(crossover=-1)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(crossover=1.2)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(mutation_rate=-1)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(mutation_rate=1.2)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(mutation_amount=-1)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(mutation_amount=3.2)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(scaling=-1)
        with self.assertRaises(ValueError):
            GeneticAlgorithm(scaling=5.2)

    def test_scale_results_with_default_scaling(self):
        ga = GeneticAlgorithm()
        results = [1, 2, 3, 4, 5]

        scaled_results = ga._scale_results(results)
        rounded_results = [round(r, 3) for r in scaled_results]
        self.assertEqual(rounded_results, [0, 0.25, 0.5, 0.75, 1])

    def test_scale_results_with_0_scaling(self):
        ga = GeneticAlgorithm(scaling=0)
        results = [1, 2, 3, 4, 5]

        scaled_results = ga._scale_results(results)
        self.assertEqual(scaled_results, [1, 1, 1, 1, 1])

    def test_scale_results_with_2_scaling(self):
        ga = GeneticAlgorithm(scaling=2)
        results = [1, 2, 3, 4, 5]

        scaled_results = ga._scale_results(results)
        rounded_results = [round(r, 3) for r in scaled_results]
        self.assertEqual(rounded_results, [0.0, 0.063, 0.25, 0.563, 1.0])

    def test_roulette(self):
        ga = GeneticAlgorithm()

        results = [1, 2, 3, 4, 5]
        np.random.seed(0)
        parent = ga._roulette(results)
        self.assertEqual(parent, 3)

        np.random.seed(50000)
        parent = ga._roulette(results)
        self.assertEqual(parent, 4)

    def test_create_child(self):
        ga = GeneticAlgorithm(crossover=0.5, mutation_rate=0.1, mutation_amount=0.2)

        np.random.seed(0)

        father = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        mother = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        child = ga._create_child(father, mother)

        self.assertEqual(child, [1, 1, 2, 1, 1, 1, 1, 2.1775452930981705, 2, 1])


class TestHillClimbing(unittest.TestCase):
    def test_valid_init(self):
        HillClimbing()
        HillClimbing(FixedDecay())

    def test_is_new_best_with_best_weight_none(self):
        hc = HillClimbing()
        self.assertEqual(hc.best_weights, None)
        self.assertEqual(hc.best_score, None)
        self.assertFalse(hc.accept_equal)
        self.assertTrue(hc.is_new_best(5))

        hc.best_score = 5
        hc.best_weights = [1, 2]

        self.assertFalse(hc.is_new_best(2))
        self.assertFalse(hc.is_new_best(5))
        self.assertTrue(hc.is_new_best(10))

        hc.accept_equal = True
        self.assertFalse(hc.is_new_best(2))
        self.assertTrue(hc.is_new_best(5))
        self.assertTrue(hc.is_new_best(10))


class TestSimulatedAnnealing(unittest.TestCase):
    def test_valid_init(self):
        SimulatedAnnealing()
        SimulatedAnnealing(temperature=FixedDecay(10, decay=0.997, minimum=0.1),
                           shift=FixedDecay(1, decay=0.995, minimum=0.01))

    def test_invalid_init(self):
        with self.assertRaises(TypeError):
            SimulatedAnnealing(temperature=1)
        with self.assertRaises(TypeError):
            SimulatedAnnealing(shift=1)
