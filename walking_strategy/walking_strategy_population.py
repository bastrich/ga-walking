from walking_strategy.walking_strategy import WalkingStrategy


class WalkingStrategyPopulation:
    def __init__(self, walking_strategies=None, size=None, initial_generation='perlin'):
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        if size is not None:
            self.walking_strategies = [WalkingStrategy(initial_generation=initial_generation) for _ in range(size)]
            return

        raise Exception('Wrong arguments')