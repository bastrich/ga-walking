import pickle
import matplotlib.pyplot as plt

x = [i+1 for i in range(50)]

with open('../results/analytics_population_2d', 'rb') as file:
    y1 = [analytics['fitness'] for analytics in pickle.load(file)][:50]
with open('../results/analytics_no_latent_mutation', 'rb') as file:
    y2 = [analytics['fitness'] for analytics in pickle.load(file)]

plt.plot(x, y1, label='latent space')
plt.plot(x, y2, label='no latent space')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.xticks([1] + list(range(5, 51, 5)))
plt.legend()
plt.show()