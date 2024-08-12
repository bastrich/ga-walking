import pickle
import matplotlib.pyplot as plt

with open('../results/analytics_population_3d', 'rb') as file:
    analytics = [analytics['types_distribution'] for analytics in pickle.load(file)]

x = [i + 1 for i in range(len(analytics))]

y = {}
for generation in analytics:
    for key, value in generation.items():
        if key not in y:
            y[key] = []
        y[key].append(value)

plt.stackplot(x, y.values(), labels=y.keys(), alpha=0.8)
plt.xlabel("Generation")
plt.ylabel("Number of muscles")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()