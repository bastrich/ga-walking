import pickle
import matplotlib.pyplot as plt

def build_plot(file_path, xticks):
    with open(file_path, 'rb') as file:
        y = [analytics['fitness'] for analytics in pickle.load(file)]
    x = [i + 1 for i in range(len(y))]

    plt.plot(x, y, label='interpolation')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xticks(xticks)
    plt.show()

build_plot('../results/analytics_population_2d', [1] + list(range(50, 301, 50)))
build_plot('../results/analytics_population_3d', [1] + list(range(500, 2001, 500)))
