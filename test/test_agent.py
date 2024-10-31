# test/test_agent.py

import unittest
import numpy as np
from src import bandit

class TestAgent(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            agt = bandit.Agent(set())
        k = 2
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        self.assertEqual(agt._k, len(agt._actions))
        self.assertEqual(agt._k, len(agt._G_t))
        self.assertEqual(agt._k, len(agt._N_t))
        self.assertGreaterEqual(agt._t_fixed, agt._k)

    def test__rewardFunc(self):
        k = 3
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        outcome = np.random.random()
        self.assertEqual(outcome, agt._rewardFunc(outcome))
        self.assertEqual(outcome, agt._rewardFunc(outcome, 'any state'))

    def test__needRandom_getWeight_getGnU(self):
        k = 4
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        outcome = np.random.random()
        for t in range(agt._t_fixed):
            self.assertTrue(agt._needRandom())
            action = agt.takeAction()
            agt.observe(action, outcome)
        self.assertFalse(agt._needRandom())
        wl = []
        for i in range(k):
            action = i + 1
            w = agt._getWeight(action)
            self.assertEqual(1.0 / agt._N_t[i], w)
            wl.append(w)
        self.assertGreater(max(wl), min(wl))
        g = agt._getGnU()
        self.assertTrue(np.all(agt._G_t == g))

    def test__takeRandomAction(self):
        k = 5
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        actions = set()
        for i in range(100):
            actions.add(agt._takeRandomAction())
        self.assertGreater(max(actions), min(actions))

    def test__update_G_t(self):
        k = 6
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        for i in range(k):
            action = i + 1
            agt._N_t[i] = 1
            agt._reward_t = 10.0
            self.assertEqual(10.0, agt._update_G_t(action))
        for i in range(k):
            action = i + 1
            agt._N_t[i] = 10
            agt._G_t[i] = 9.0
            self.assertEqual(9.1, agt._update_G_t(action))

    def test_observe(self):
        k = 7
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        for t in range(100):
            tm1 = agt._t
            Nm1 = agt._N_t.copy()
            Gm1 = agt._G_t.copy()
            action = agt.takeAction()
            i = action - 1
            outcome = np.random.random()
            agt.observe(action, outcome)
            self.assertEqual(agt._t, tm1 + 1)
            self.assertEqual(agt._reward_t, outcome)
            self.assertEqual(agt._N_t[i], Nm1[i] + 1)
            self.assertNotEqual(agt._G_t[i], Gm1[i])

    def test_takeAction(self):
        k = 8
        actionSet = set(range(1, k + 1))
        agt = bandit.Agent(actionSet)
        bias = np.zeros(k)
        bias[5] += 1.0
        for t in range(30 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(5, np.argmax(agt._N_t))

class TestEpsilonGreedy(unittest.TestCase):
    def test_init(self):
        k = 2
        actionSet = set(range(1, k + 1))
        with self.assertRaises(ValueError):
            agt = bandit.EpsilonGreedy(actionSet, 1.1)
        with self.assertRaises(ValueError):
            agt = bandit.EpsilonGreedy(actionSet, -0.1)

    def test_takeAction(self):
        k = 3
        actionSet = set(range(1, k + 1))
        agt = bandit.EpsilonGreedy(actionSet, 0.1)
        bias = np.zeros(k)
        bias[0] += 1.0
        for t in range(30 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(0, np.argmax(agt._N_t))
        bias[0] -= 1.0
        bias[1] += 2.0
        for t in range(120 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(1, np.argmax(agt._N_t))

class TestDecayingEpsilonGreedy(unittest.TestCase):
    def test_init(self):
        k = 2
        actionSet = set(range(1, k + 1))
        with self.assertRaises(ValueError):
            agt = bandit.DecayingEpsilonGreedy(actionSet, 0.1, 0.0)
        with self.assertRaises(ValueError):
            agt = bandit.DecayingEpsilonGreedy(actionSet, 0.1, 1.1)

    def test_takeAction(self):
        k = 3
        actionSet = set(range(1, k + 1))
        agt = bandit.DecayingEpsilonGreedy(actionSet, 0.1, 0.1)
        bias = np.zeros(k)
        bias[0] += 1.0
        for t in range(30 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(0, np.argmax(agt._N_t))
        bias[0] -= 1.0
        bias[1] += 2.0
        for t in range(150 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(1, np.argmax(agt._N_t))

class TestUCB(unittest.TestCase):
    def test_takeAction_gatName(self):
        k = 3
        actionSet = set(range(1, k + 1))
        agt = bandit.UCB(actionSet, 2.0)
        bias = np.zeros(k)
        bias[0] += 1.0
        for t in range(15 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(0, np.argmax(agt._N_t))
        bias[0] -= 1.0
        bias[1] += 2.0
        for t in range(30 * k):
            action = agt.takeAction()
            i = action - 1
            outcome = bias[i] + np.random.random()
            agt.observe(action, outcome)
        self.assertEqual(1, np.argmax(agt._N_t))
        self.assertEqual('UCB c=2.0', agt.getName())

if __name__ == '__main__':
    unittest.main()
