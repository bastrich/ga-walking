import numpy as np

class WalkingStrategy:
    def __init__(self, period, muscle_activations_fourier_coefficients=None):
        self.period = period

        if muscle_activations_fourier_coefficients is None:
            self.muscle_activations_fourier_coefficients = [WalkingStrategy.generate_single_muscle_activation_fourier_coefficients() for _ in range(11)]
        else:
            self.muscle_activations_fourier_coefficients = muscle_activations_fourier_coefficients

        self.muscle_activations_cache = np.vstack([self.normalize(self.calculate_fourier_series_sums_1(coefficients)) for coefficients in self.muscle_activations_fourier_coefficients] +
         [self.normalize(self.calculate_fourier_series_sums_2(coefficients)) for coefficients in self.muscle_activations_fourier_coefficients]).transpose()

    def normalize(self, muscle_activations):
        return muscle_activations

        # min_value = np.min(muscle_activations)
        # max_value = np.max(muscle_activations)
        #
        # if min_value == max_value:
        #     return muscle_activations
        #
        # return (muscle_activations - min_value) / (max_value - min_value)

    def calculate_fourier_series_sums_1(self, single_muscle_coefficients):
        return np.array([self.calculate_fourier_series_sum(single_muscle_coefficients, i) for i in range(self.period)])

    def calculate_fourier_series_sums_2(self, single_muscle_coefficients):
        return np.array([self.calculate_fourier_series_sum(single_muscle_coefficients, i + self.period / 2) for i in range(self.period)])

    @staticmethod
    def generate_single_muscle_activation_fourier_coefficients():
        a = [
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101)
        ]
        b = [
            0,
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101),
            np.random.randint(-100, 101)
        ]
        phases = [
            np.random.randint(-314, 315),
            np.random.randint(-314, 315),
            np.random.randint(-314, 315),
            np.random.randint(-314, 315),
            np.random.randint(-314, 315)
        ]

        return [a[0], a[1], b[1], phases[0], a[2], b[2], phases[1], a[3], b[3], phases[2], a[4], b[4], phases[3], a[5], b[5], phases[4]]

    def get_muscle_activations(self, time):
        return self.muscle_activations_cache[time % self.period]

    def calculate_fourier_series_sum(self, coefficients, time):
        return (coefficients[0] / 100 +
                coefficients[1] / 100 * np.cos(2 * np.pi * time / self.period + coefficients[3] / 100) + coefficients[2] / 100 * np.sin(2 * np.pi * time / self.period + coefficients[3] / 100) +
                coefficients[4] / 100 * np.cos(2 * 2 * np.pi * time / self.period + coefficients[6] / 100) + coefficients[5] / 100 * np.sin(2 * 2 * np.pi * time / self.period + coefficients[6] / 100) +
                coefficients[7] / 100 * np.cos(3 * 2 * np.pi * time / self.period + coefficients[9] / 100) + coefficients[8] / 100 * np.sin(3 * 2 * np.pi * time / self.period + coefficients[9] / 100) +
                coefficients[10] / 100 * np.cos(4 * 2 * np.pi * time / self.period + coefficients[12] / 100) + coefficients[11] / 100 * np.sin(4 * 2 * np.pi * time / self.period + coefficients[12] / 100) +
                coefficients[13] / 100 * np.cos(5 * 2 * np.pi * time / self.period + coefficients[15] / 100) + coefficients[14] / 100 * np.sin(5 * 2 * np.pi * time / self.period + coefficients[15] / 100))
