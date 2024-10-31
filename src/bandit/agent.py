import numpy as np

class Agent:
    """
    The agent that interacts with the environment by taking actions and observing outcomes. It has its own reward function to evaluate the 
    reward based on the outcome. The agent's goal is to maximize the total rewards generated by its actions over time.
    The agent is greedy, which only explores during a fixed time period and then always exploits.
    """
    def __init__(self, actions: set[int]) -> None:
        self._actions = actions
        self._name = 'greedy'

        self.reset()
        if self._k == 0:
            raise ValueError('actions should not be an empty set.')

    def _rewardFunc(self, outcome: float, state: str = None) -> float:
        return outcome

    def _getWeight(self, action: int) -> float:
        return 1.0 / self._N_t[action - 1]
    
    def _update_G_t(self, action: int) -> float:
        i = action - 1
        w = self._getWeight(action)
        return self._reward_t if self._G_t[i] == -np.inf else (1 - w) * self._G_t[i] + w * self._reward_t

    def _needRandom(self) -> bool:
        return True if self._t < self._t_fixed else False
    
    def _takeRandomAction(self) -> int:
        a = (np.random.uniform(size=1)*self._k).astype(int)
        return a[0] + 1
    
    def _getGnU(self) -> np.ndarray:
        return self._G_t

    def takeAction(self) -> int:
        if self._needRandom():
            return self._takeRandomAction()
        return np.argmax(self._getGnU()) + 1

    def observe(self, action: int, outcome: float, state: str = None) -> None:
        i = action - 1
        self._N_t[i] += 1
        self._reward_t = self._rewardFunc(outcome, state)
        self._G_t[i] = self._update_G_t(action)
        self._t += 1
        if state != None:
            self._state_t = state

    def reset(self) -> None:
        self._t = 0
        self._k = len(self._actions)
        self._t_fixed = 15 * self._k
        self._state_t = 's0'
        self._reward_t = None
        self._G_t = np.full(self._k, -np.inf)
        self._N_t = np.zeros(self._k).astype(int)
    
    def getName(self) -> str:
        return self._name

class EpsilonGreedy(Agent):
    """
    The epsilon greedy agent
    """
    def __init__(self, actions: set[int], epsilon: float):
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError('epsilon must be within [0.0, 1.0]')
        
        super().__init__(actions)
        self._epsilon = epsilon
        self._name = f'$\\epsilon$-greedy $\\epsilon$={epsilon}'

    def _needRandom(self) -> bool:
        rn = np.random.uniform(size=1)[0]
        return True if rn < self._epsilon else False

class DecayingEpsilonGreedy(EpsilonGreedy):
    """
    The decaying epsilon greedy agent
    """
    def __init__(self, actions: set[int], epsilon: float, alpha: float):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError('alpha must be within (0.0, 1.0]')

        super().__init__(actions, epsilon)
        self._alpha = alpha
        self._name = f'decaying $\\epsilon$-greedy $\\epsilon$={epsilon}, $\\alpha$={alpha}'

    def _getWeight(self, action: int) -> float:
        return self._alpha

class UCB(Agent):
    """
    The UCB agent
    """
    def __init__(self, actions: set[int], c: float) -> None:
        super().__init__(actions)
        self._c = c
        self._name = f'UCB c={c}'

    def _takeRandomAction(self) -> int:
        return self._randomActions[self._t]
    
    def _getGnU(self) -> np.ndarray:
        return self._G_t + self._U_t

    def observe(self, action: int, outcome: float, state = None) -> None:
        super().observe(action, outcome, state)
        for i in range(self._k):
            self._U_t[i] = np.inf if self._N_t[i] == 0 else self._c * np.sqrt((np.log(self._t) / self._N_t[i]))

    def reset(self) -> None:
        super().reset()
        self._U_t = np.full(self._k, np.inf)
        self._t_fixed = self._k
        self._randomActions = np.arange(self._k)
        np.random.shuffle(self._randomActions)

if __name__ == '__main__':
    print('only runs in the top-level')