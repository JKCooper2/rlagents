import unittest

from rlagents.memory import ListMemory


class TestListMemory(unittest.TestCase):
    def test_new_string(self):
        lm = ListMemory()
        lm.new("observation")

        self.assertTrue("observation" in lm.items)

    def test_new_list(self):
        lm = ListMemory()
        lm.new(["observation", "reward", "action"])

        self.assertTrue("observation" in lm.items)
        self.assertTrue("reward" in lm.items)
        self.assertTrue("action" in lm.items)
