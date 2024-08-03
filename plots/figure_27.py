import pickle
import matplotlib.pyplot as plt

x = [i+1 for i in range(10)]

with open('../results/fitness_no_perlin', 'rb') as file:
    y1 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/fitness_perlin', 'rb') as file:
    y2 = [analytics['fitness'] for analytics in pickle.load(file)]

plt.plot(x, y1, label="Total random")
plt.plot(x, y2, label="Perlin noise")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()