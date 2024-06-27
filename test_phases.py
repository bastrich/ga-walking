import pickle

import matplotlib.pyplot as plt

from perlin_noise import PerlinNoise

from walking_strategy import WalkingStrategy

import numpy as np

import time

# create data
x = [i for i in range(200)]

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

y = best_walking_strategy.muscles[1].muscle_activations

# from muscle import Muscle
#
# muscle = Muscle(200)
# y1 = [muscle.get_muscle_activation(i) for i in range(200)]
# for i in range(10):
#     muscle = muscle.mutate(0.5, 1.5)
# y2 = [muscle.get_muscle_activation(i) for i in range(200)]


# plot lines
plt.plot(x, y, label="original", lw = 2)
# plt.plot(x, y2, label="mutated", lw = 1.5)
plt.legend()
plt.show()