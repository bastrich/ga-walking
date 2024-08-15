import pickle
import matplotlib.pyplot as plt
import numpy as np


def build_plot(file_path, xticks):
    with open(file_path, 'rb') as file:
        y = [np.round(analytics['energy'] / abs(analytics['distance']), 2) for analytics in pickle.load(file)]
    x = [i+1 for i in range(len(y))]

    plt.plot(x, y)
    plt.xlabel("Generation")
    plt.ylabel("Energy per meter, conventional units")
    plt.xticks(xticks)
    plt.show()

build_plot('../results/analytics_population_2d', [1] + list(range(100, 301, 100)))
build_plot('../results/analytics_population_3d', [1] + list(range(200, 2001, 200)))