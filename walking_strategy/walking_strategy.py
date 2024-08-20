import numpy as np
from walking_strategy.muscle import Muscle


class WalkingStrategy:
    """
    Represents a walking strategy with muscles.
    """

    def __init__(self, mode=None, fixed_period=None, fixed_type=None, fixed_sampling_interval=None, fixed_precision=None, initial_generation='perlin', frame_skipping='action_repeat', muscles=None):
        """
        Initiates a new instance of walking strategy.

        Args:
            mode (str):
            fixed_period (int):
            fixed_type (int):
            fixed_sampling_interval (int):
            fixed_precision (int):
            initial_generation (str):
            frame_skipping (str):
            muscles (list):
        """
        if muscles is not None:
            if len(muscles) != 11:
                raise ValueError("muscles must have size 11")

            if np.unique([muscle.period for muscle in muscles]).size != 1:
                raise ValueError("muscles must have the same period")

            if fixed_period is not None and fixed_period != muscles[0].period:
                raise ValueError("fixed_period should be the same as muscles period")

            if fixed_type is not None and (np.unique([muscle.type for muscle in muscles]).size != 1 or fixed_type != muscles[0].type):
                raise ValueError("fixed_type should be the same as type of all muscles")

            if fixed_sampling_interval is not None and (np.unique([muscle.sampling_interval for muscle in muscles]).size != 1 or fixed_sampling_interval != muscles[0].sampling_interval):
                raise ValueError("fixed_sampling_interval should be the same as sampling_interval of all muscles")

            if fixed_precision is not None and (np.unique([muscle.precision for muscle in muscles]).size != 1 or fixed_precision != muscles[0].precision):
                raise ValueError("fixed_precision should be the same as precision of all muscles")

            self.period = muscles[0].period
            self.muscles = muscles
        elif fixed_period is not None:
            self.period = fixed_period
            self.muscles = [Muscle(period=self.period, mode=mode, type=fixed_type, sampling_interval=fixed_sampling_interval, precision=fixed_precision, initial_generation=initial_generation, frame_skipping=frame_skipping) for _ in range(11)]
        else:
            self.period = np.random.choice(Muscle.PERIODS)
            self.muscles = [Muscle(period=self.period, mode=mode, type=fixed_type, sampling_interval=fixed_sampling_interval, precision=fixed_precision, initial_generation=initial_generation, frame_skipping=frame_skipping) for _ in range(11)]

        self.fixed_period = fixed_period
        self.fixed_type = fixed_type
        self.fixed_sampling_interval = fixed_sampling_interval
        self.fixed_precision = fixed_precision

    def get_muscles_activations(self, time):
        return ([muscle.get_activation(time) for muscle in self.muscles] + [muscle.get_activation(time + self.period // 2) for muscle in self.muscles])

    def mutate(self, mutation_rate, period_mutation_rate, type_mutation_rate, sampling_interval_mutation_rate, precision_mutation_rate, components_mutation_rate, components_mutation_amount):
        if np.random.uniform() > mutation_rate:
            return self

        new_muscles = self.muscles
        if self.fixed_period is None and np.random.uniform() <= period_mutation_rate:
            new_period = np.random.choice(Muscle.PERIODS)
            new_muscles = [muscle.with_period(new_period) for muscle in new_muscles]

        if self.fixed_type is None:
            new_muscles = [muscle.mutate_type() if np.random.uniform() <= type_mutation_rate else muscle for muscle in new_muscles]

        if self.fixed_sampling_interval is None:
            new_muscles = [muscle.mutate_sampling_interval() if np.random.uniform() <= sampling_interval_mutation_rate else muscle for muscle in new_muscles]

        if self.fixed_precision is None:
            new_muscles = [muscle.mutate_precision() if np.random.uniform() <= precision_mutation_rate else muscle for muscle in new_muscles]

        new_muscles = [muscle.mutate_components(components_mutation_rate, components_mutation_amount) for muscle in new_muscles]

        return WalkingStrategy(
            fixed_period=self.fixed_period,
            fixed_type=self.fixed_type,
            fixed_sampling_interval=self.fixed_sampling_interval,
            fixed_precision=self.fixed_precision,
            muscles=new_muscles
        )

    def crossover(self, other):
        new_period = np.random.choice([self.period, other.period])
        switch_index = np.random.randint(0, len(self.muscles) + 1)

        new_muscles = []
        for i in range(len(self.muscles)):
            if i < switch_index:
                new_muscles.append(self.muscles[i].with_period(new_period))
            else:
                new_muscles.append(other.muscles[i].with_period(new_period))

        return WalkingStrategy(
            fixed_period=self.fixed_period,
            fixed_type=self.fixed_type,
            fixed_sampling_interval=self.fixed_sampling_interval,
            fixed_precision=self.fixed_precision,
            muscles=new_muscles
        )
