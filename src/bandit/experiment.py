import matplotlib.pyplot as plt
import numpy as np
from .agent import *
from .environment import *

class Experiment:
    """
    An experiment picks an environment and one or more agents and compare their performances within a period of time.
    """
    def __init__(self, times: int, steps: int, **kwargs) -> None:
        self._kwargs = {'k': 2, 'dist': None, 'smooth': False,
                        'agents': [{'algorithm': 'greedy'}, 
                                   {'algorithm': 'epsilon-greedy', 'epsilon': 0.1}, 
                                   {'algorithm': 'decaying', 'epsilon': 0.1, 'alpha': 0.1}, 
                                   {'algorithm': 'UCB', 'c': 2.0}]}
        for k, v in kwargs.items():
            self._kwargs[k] = v

        self._times = times
        self._steps = steps
        k = self._kwargs['k']
        dist = self._kwargs['dist']
        self._env = Environment(k) if dist == None else Environment(k, dist)

        self._agents = []
        for agt_dict in self._kwargs['agents']:
            if agt_dict['algorithm'] == 'greedy':
                self._agents.append(Agent(self._env.getActionSet()))
            if agt_dict['algorithm'] == 'epsilon-greedy':
                self._agents.append(EpsilonGreedy(self._env.getActionSet(), agt_dict['epsilon']))
            if agt_dict['algorithm'] == 'decaying':
                self._agents.append(DecayingEpsilonGreedy(self._env.getActionSet(), agt_dict['epsilon'], agt_dict['alpha']))
            if agt_dict['algorithm'] == 'UCB':
                self._agents.append(UCB(self._env.getActionSet(), agt_dict['c']))

    def run(self) -> None:
        self._dataset = []
        for i in range(len(self._agents)):
            agt = self._agents[i]
            avg_n = np.zeros(self._steps)
            avg_t = np.zeros(self._steps)
            for n in range(self._times):
                for t in range(self._steps):
                    action = agt.takeAction()
                    outcome = self._env.executeAction(action)
                    agt.observe(action, outcome)
                    if t == 0 or not self._kwargs['smooth']:
                        avg_t[t] = outcome
                    else:
                        avg_t[t] = (1.0 - 1.0 / (t + 1)) * avg_t[t - 1] + (1.0 / (t + 1)) * outcome
                for t in range(self._steps):
                    avg_n[t] = (1.0 - 1.0 / (n + 1)) * avg_n[t] + (1.0 / (n + 1)) * avg_t[t]
                agt.reset()
            self._dataset.append(avg_n)

    def plot(self, path: str) -> None:
        params = self._env.getParam()
        plt.title(f"Probability distribution: {params['dist_name']}, k = {self._kwargs['k']}")
        plt.xlabel('Steps')
        if self._kwargs['smooth']:
            y_reward = '$\\frac{total\\ rewards}{steps}$'
        else:
            y_reward = 'reward'
        plt.ylabel(f"Average {y_reward}")
        v_star = np.full(self._steps, max(params['mean']))
        plt.plot(v_star, label = '$v_\\ast$')
        for i in range(len(self._agents)):
            agt = self._agents[i]
            plt.plot(self._dataset[i], label = agt.getName())
        plt.legend()
        plt.savefig(path, format = 'svg')
