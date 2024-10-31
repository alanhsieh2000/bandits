# test/test_experiment.py

import unittest
import numpy as np
import scipy as sp
import os.path
from src import bandit

class TestExperiment(unittest.TestCase):
    def test_run(self):
        config = {'k': 3, 'dist': sp.stats.norm}
        expt = bandit.Experiment(2, 1000, **config)
        expt.run()
        data = expt._dataset
        self.assertEqual(4, len(data))
        self.assertEqual(1000, len(data[0]))

    def test_plot(self):
        config = {'k': 10, 'dist': sp.stats.norm, 'smooth': True, 
                  'agents': [{'algorithm': 'greedy'}, 
                             {'algorithm': 'epsilon-greedy', 'epsilon': 0.01}, 
                             {'algorithm': 'epsilon-greedy', 'epsilon': 0.1}, 
                             {'algorithm': 'UCB', 'c': 2}]}
        times = 2
        expt = bandit.Experiment(times, 1000, **config)
        expt.run()
        path_parts = ['experiment', f"{times}"]
        if config['smooth']:
            path_parts.insert(1, 'smooth')
        path = f"experiment-{times}.svg"
        path = '-'.join(path_parts) + '.svg'
        self.assertFalse(os.path.isdir(path))
        expt.plot(path)
        self.assertTrue(os.path.isfile(path))

if __name__ == '__main__':
    unittest.main()
