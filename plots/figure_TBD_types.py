import pickle
import matplotlib.pyplot as plt
import numpy as np

x = [i+1 for i in range(50)]

with open('../results/analytics_all_types', 'rb') as file:
    y1 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_direct', 'rb') as file:
    y2 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_fourier', 'rb') as file:
    y3 = [analytics['fitness'] for analytics in pickle.load(file)]


plt.plot(x, y1, label='all types')
plt.plot(x, y2, label='direct')
plt.plot(x, y3, label='fourier')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.xticks([1] + list(range(5, 51, 5)))
plt.legend()
plt.show()