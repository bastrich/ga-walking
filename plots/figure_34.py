import pickle
import matplotlib.pyplot as plt

x = [i+1 for i in range(50)]

with open('../results/analytics_fitness_function_from_design', 'rb') as file:
    y1 = [analytics['distance'] for analytics in pickle.load(file)]
with open('../results/analytics_fitness_function_improved', 'rb') as file:
    y2 = [analytics['distance'] for analytics in pickle.load(file)]

plt.plot(x, y1, label='fitness function from Design')
plt.plot(x, y2, label='improved fitness function')
plt.xlabel("Generation")
plt.ylabel("Distance")
plt.xticks([1] + list(range(5, 51, 5)))
plt.legend()
plt.show()