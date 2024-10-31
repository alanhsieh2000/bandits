import numpy as np
import scipy as sp

class Environment:
    """
    The environment that has an action set and generates an undeterministic outcome after an action executed.
    """
    UNIFORM = 'uniform_gen'
    NORM = 'norm_gen'
    BETA = 'beta_gen'

    def __init__(self, k: int, dist=sp.stats.uniform) -> None:
        if k <= 0:
            raise ValueError('k must be greater than 0.')
            
        self._k = k
        self._dist = dist
        self._dist_name = type(dist).__name__

        if self._dist_name not in {self.UNIFORM, self.BETA, self.NORM}:
            raise ValueError('dist is not a supported probability distribution.')
        
        if self._dist_name == self.BETA:
            rn = sp.stats.uniform.rvs(size=k)
            self._a = (rn * 100).astype(int)
            self._b = 100 - self._a
            self._loc = np.zeros(k)
            self._scale = np.ones(k)
        else:
            rn = sp.stats.uniform.rvs(loc=-2.0, scale=4.0, size=k)
            self._loc = rn
            self._scale = np.ones(k)

    def getActionSet(self) -> set[int]:
        return set(range(1, self._k + 1))
    
    def executeAction(self, action: int) -> float:
        i = action - 1
        keywords = {'loc': self._loc[i], 'scale': self._scale[i], 'size': 1}
        if self._dist_name == self.BETA:
            keywords['a'] = self._a[i]
            keywords['b'] = self._b[i]
        return self._dist.rvs(**keywords)[0]

    def getParam(self) -> dict[str, str | np.ndarray]:
        params = {'dist_name': self._dist_name}
        if self._dist_name == self.UNIFORM:
            params['mean'] = self._loc + self._scale / 2
        elif self._dist_name == self.NORM:
            params['mean'] = self._loc
        elif self._dist_name == self.BETA:
            params['mean'] = self._a / (self._a + self._b)
        return params

if __name__ == '__main__':
    print('only runs in the top-level')