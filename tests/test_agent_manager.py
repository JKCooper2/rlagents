import unittest
import json

from rlagents.examples.agents import crossentropy_discretelinear


class TestAgentManager(unittest.TestCase):
    def test_export(self):
        # Dictionary is unordered, need a more thorough check

        am = crossentropy_discretelinear()

        check = "{'Agent Manager': {'Times Run': 1, 'Evolution': {'Elite': 0.2, 'Type': 'CrossEntropy'}, 'Agents': {'Agent 0': {'Observation FA': {'Type': 'Default'}, 'Action FA': {'Type': 'Discrete Max'}, 'Optimiser': {'Type': 'Default'}, 'Exploration': {'Type': 'Default'}, 'Memory': {'Type': 'List Memory', 'Columns': ['rewards', 'done', 'observations', 'actions'], 'Size': 2}, 'Model': {'Normalise': False, 'Bias': True, 'Type': 'Weighted Linear Model'}}, 'Agent 1': {'Observation FA': {'Type': 'Default'}, 'Action FA': {'Type': 'Discrete Max'}, 'Optimiser': {'Type': 'Default'}, 'Exploration': {'Type': 'Default'}, 'Memory': {'Type': 'List Memory', 'Columns': ['rewards', 'done', 'observations', 'actions'], 'Size': 2}, 'Model': {'Normalise': False, 'Bias': True, 'Type': 'Weighted Linear Model'}}}}}}"

        # self.assertEqual(json.dumps(am.export()), check)