import pickle
import matplotlib.pyplot as plt

x = [i+1 for i in range(300)]
#
with open('../results/analytics_interpolation', 'rb') as file:
    y1 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_population_2d', 'rb') as file:
    y2 = [analytics['fitness'] for analytics in pickle.load(file)]

plt.plot(x, y1, label='interpolation')
plt.plot(x, y2, label='action_repeat')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.xticks([1] + list(range(50, 301, 50)))
plt.legend()
plt.show()