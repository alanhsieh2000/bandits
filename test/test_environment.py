# test/test_environment.py

import unittest
import scipy as sp
from src import bandit

class TestEnvironment(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            env = bandit.Environment(0)
        with self.assertRaises(ValueError):
            env = bandit.Environment(1, sp.stats.lognorm)
        envNorm = bandit.Environment(2, sp.stats.norm)
        envBeta = bandit.Environment(3, sp.stats.beta)

    def test_getActionSet(self):
        k = 4
        env = bandit.Environment(k)
        actions = env.getActionSet()
        self.assertIsInstance(actions, set)
        self.assertEqual(len(actions), k)
        self.assertEqual(max(actions), k)
        self.assertEqual(min(actions), 1)
        self.assertIsInstance(actions.pop(), int)

    def test_executeAction(self):
        k = 5
        env = bandit.Environment(k)
        actions = env.getActionSet()
        for a in actions:
            o = env.executeAction(a)
            self.assertIsInstance(o, float)

    def test_getParam(self):
        k = 6
        env = bandit.Environment(k)
        params = env.getParam()
        self.assertIsInstance(params, dict)
        self.assertIn('dist_name', params)
        self.assertIn('mean', params)
        self.assertEqual(len(params['mean']), k)

if __name__ == '__main__':
    unittest.main()
