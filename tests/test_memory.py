import unittest

from rlagents.memory import ListMemory, PandasMemory


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


class TestPandasMemory(unittest.TestCase):
    def test_new_string(self):
        pm = PandasMemory()
        pm.new("observation")

        self.assertTrue("observation" in pm.df.columns.values)

    def test_new_list(self):
        pm = PandasMemory()
        pm.new(["observation", "actions"])

        self.assertTrue("observation" in pm.df.columns.values)
        self.assertTrue("actions" in pm.df.columns.values)

    def test_store_item(self):
        pm = PandasMemory()
        pm.new(["observation", "actions"])

        pm.store({"observation": 1})

        self.assertEqual(pm.df['observation'][0], 1)

    def test_store_list_items(self):
        pm = PandasMemory()
        pm.new(["observation", "action"])

        pm.store([{"observation": 1, "action": 1}, {"observation": 2, "action": 0}])

        self.assertEqual(pm.df['observation'][0], 1)
        self.assertEqual(pm.df['action'][0], 1)
        self.assertEqual(pm.df['observation'][1], 2)
        self.assertEqual(pm.df['action'][1], 0)

    def test_fetch_last_name_pandas(self):
        pm = PandasMemory()
        pm.new(["observation", "action"])
        pm.store([{"observation": 1, "action": 1}, {"observation": 2, "action": 0}])

        fetch = pm.fetch_last(2, 'observation')

        self.assertEqual(fetch[0], 1)
        self.assertEqual(fetch[1], 2)

    def test_fetch_last_pandas(self):
        pm = PandasMemory()
        pm.new(["observation", "action"])
        pm.store([{"observation": 1, "action": 1}, {"observation": 2, "action": 0}])

        fetch = pm.fetch_last(2)

        self.assertEqual(fetch['observation'][0], 1)
        self.assertEqual(fetch['observation'][1], 2)
        self.assertEqual(fetch['action'][0], 1)
        self.assertEqual(fetch['action'][1], 0)

    def test_count_name(self):
        pm = PandasMemory()
        pm.new(["observation", "action"])
        pm.store([{"observation": 1, "action": 1}, {"observation": 2, "action": 0}])

        mem = pm.count('observation')

        self.assertEqual(mem, 2)

    def test_count_name_empty(self):
        pm = PandasMemory()
        pm.new(["observation", "action"])
        mem = pm.count('observation')

        self.assertEqual(mem, 0)

    def test_count_no_name(self):
        pm = PandasMemory()
        pm.new(["observation", "action"])
        pm.store([{"observation": 1, "action": 1}, {"observation": 2, "action": 0}])
        mem = pm.count()

        self.assertEqual(mem, 2)