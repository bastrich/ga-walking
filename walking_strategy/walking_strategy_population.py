from walking_strategy.walking_strategy import WalkingStrategy


class WalkingStrategyPopulation:
    def __init__(self, **kwargs):
        walking_strategies = kwargs.get('walking_strategies')
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        size = kwargs.get('size')
        if size is not None:
            self.walking_strategies = [WalkingStrategy() for _ in range(size)]
            return

        raise Exception('Wrong arguments')