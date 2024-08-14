import pickle
import matplotlib.pyplot as plt

x = [i+1 for i in range(20)]

with open('../results/analytics_elitism_0_1', 'rb') as file:
    y1 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_elitism_0_15', 'rb') as file:
    y2 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_elitism_0_2', 'rb') as file:
    y3 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_elitism_0_25', 'rb') as file:
    y4 = [analytics['fitness'] for analytics in pickle.load(file)]
with open('../results/analytics_elitism_0_3', 'rb') as file:
    y5 = [analytics['fitness'] for analytics in pickle.load(file)]


plt.plot(x, y1, label='elitism=0.1')
plt.plot(x, y2, label='elitism=0.15')
plt.plot(x, y3, label='elitism=0.2')
plt.plot(x, y4, label='elitism=0.25')
plt.plot(x, y5, label='elitism=0.3')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.xticks(x)
plt.legend()
plt.show()