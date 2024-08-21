from walking_strategy.walking_strategy import WalkingStrategy


class WalkingStrategyPopulation:
    """
    Represents a population of walking strategies.
    """

    def __init__(self, mode=None, walking_strategies=None, size=None, initial_generation='perlin', frame_skipping='action_repeat'):
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        if size is not None:
            self.walking_strategies = [WalkingStrategy(mode=mode, fixed_period=160 if mode == '3D' else None, initial_generation=initial_generation, frame_skipping=frame_skipping) for _ in range(size)]
            return

        raise Exception('Wrong arguments')