from walking_strategy import WalkingStrategy
import numpy as np

class WalkingStrategyPopulation:
    def __init__(self, period, **kwargs):
        walking_strategies = kwargs.get('walking_strategies')
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        size = kwargs.get('size')
        if size is not None:
            self.walking_strategies = [WalkingStrategy(period) for _ in range(size)]
            return

        raise Exception('Wrong arguments')

    @staticmethod
    def get_fitness_map(fitness_values):
        fitmap = []
        total = 0
        for f in fitness_values:
            total = total + f
            fitmap.append(total)
        return fitmap

    def select_parents(self, fitmap):
        parent1_index = self.select_parent(fitmap)

        diff = fitmap[parent1_index]
        if parent1_index > 0:
            diff -= fitmap[parent1_index - 1]

        r = np.random.uniform()  # 0-1
        r = r * (fitmap[-1] - diff)

        for i in range(len(fitmap)):
            if i == parent1_index:
                continue
            if i < parent1_index and r <= fitmap[i] or i > parent1_index and r <= fitmap[i] - diff:
                return self.walking_strategies[parent1_index], self.walking_strategies[i]

        return self.walking_strategies[parent1_index], self.walking_strategies[len(fitmap) - 1]

    def select_parent(self, fitmap):
        r = np.random.rand()  # 0-1
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i

        return len(fitmap) - 1