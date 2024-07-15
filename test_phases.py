import pickle

import matplotlib.pyplot as plt

from perlin_noise import PerlinNoise

from walking_strategy import WalkingStrategy

import numpy as np

import time

# create data
x1 = [i for i in range(200)]
x2 = [i for i in range(250)]


# with open('population', 'rb') as file:
#     walking_strategy = pickle.load(file).walking_strategies[0]
# walking_strategy = WalkingStrategy(200)

# with open('best', 'rb') as file:
#     walking_strategy = pickle.load(file)

# y1 = [walking_strategy.get_muscle_activations(i)[2] for i in x]
# y2 = [walking_strategy.get_muscle_activations(i+200)[2] for i in x]

# noise = PerlinNoise(octaves=5)
#
# y1 = np.array([noise(i / 40) for i in x[:40]])
# min_value = np.min(y1)
# max_value = np.max(y1)
# if min_value != max_value:
#     y1 = (y1 - min_value) / (max_value - min_value)
#
# y_o = np.tile(y1, (5, 1)).T.flatten()
#
# original_indices = np.arange(40)
# new_indices = np.linspace(0, 39, 200)
#
# from scipy.interpolate import interp1d
# start_time = time.time_ns()
# linear_interpolator = interp1d(original_indices, y1, kind='cubic', fill_value='extrapolate')
# y_e = linear_interpolator(new_indices)
# end_time = time.time_ns()
#
# print(f'Execution time: {(end_time - start_time)} s')
#
# # y1 = np.interp(new_indices, original_indices, y1)
#
#
#
# # fft_signal = np.fft.fft(y1)
# # # fft_signal[5:] = 0
# # y2 = np.real(np.fft.ifft(fft_signal))
#
# # y2 = [noise(i / 40) for i in x]

# y = WalkingStrategy(period=200, symmetric=True).crossover(WalkingStrategy(period=200, symmetric=True)).mutate().muscle_activations[10]

with open('population', 'rb') as file:
    best_walking_strategy = pickle.load(file).walking_strategies[0]

# best_walking_strategy = best_walking_strategy.with_period(150)
#
x1 = [i for i in range(best_walking_strategy.period)]
y1 = best_walking_strategy.muscles[7].activations
# best_walking_strategy.muscles[0].change_precision(10)
# for i in range(1000):
#     best_walking_strategy.muscles[0] = best_walking_strategy.muscles[0].mutate(0.3, 0.3)

# y2 = [best_walking_strategy.muscles[0].with_period(150).get_muscle_activation(i) for i in range(150)]

# best_walking_strategy.muscles[0].change_precision(10)
# y2 = best_walking_strategy.muscles[0].activations
#
# muscle = best_walking_strategy.muscles[3]
# for i in range(100):
#     muscle = muscle.mutate(0.3, 0.3)
# y2 = [muscle.get_muscle_activation(i) for i in range(200)]

from simple_muscle import Muscle

# walking_strategy = WalkingStrategy(200)
# y1 = [walking_strategy.muscles[2].get_muscle_activation(i) for i in range(walking_strategy.period)]
# x1 = [i for i in range(walking_strategy.period)]
# for i in range(100):
#     walking_strategy.muscles[2] = walking_strategy.muscles[2].mutate(0.3, 0.3)
# # muscle = Muscle(period=200, precision=5, fourier_coefficients=[np.real(fc) + 0j for fc in muscle.fourier_coefficients])
#
# y2 = [walking_strategy.muscles[2].get_muscle_activation(i) for i in range(walking_strategy.period)]
# x2 = [i for i in range(walking_strategy.period)]

# noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=1))
# y1 = np.array([noise(i / (200 // 5)) for i in range(200 // 5)])
#
# min_value = np.min(y1)
# max_value = np.max(y1)
# if min_value != max_value:
#     y1 = (y1 - min_value) / (max_value - min_value)
#
# fft = np.fft.fft(y1)
# #
# y2 = (np.real(fft[0]) + np.array([np.sum([np.real(fft[i + 1]) * np.cos((i + 1) * 2 * np.pi / 40 * j) - np.imag(fft[i + 1]) * np.sin((i + 1) * 2 * np.pi / 40 * j) for i in range(len(fft) - 1)]) for j in range(40)])) / 40


# plot lines
plt.plot(x1, y1, label="original", lw = 2)
# plt.plot(x2, y2, label="mutated", lw = 1.5)
plt.legend()
plt.show()

