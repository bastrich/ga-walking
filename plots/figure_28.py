import pickle
import matplotlib.pyplot as plt

with open('../results/analytics_population_2d', 'rb') as file:
    analytics = pickle.load(file)

x = [i + 1 for i in range(len(analytics))]

y1 = {}
y2 = {}
y3 = {}
y4 = {}
for generation in analytics:
    for key, value in generation['periods_distribution'].items():
        if key not in y1:
            y1[key] = []
        y1[key].append(value)
    for key, value in generation['types_distribution'].items():
        if key not in y2:
            y2[key] = []
        y2[key].append(value)
    for key, value in generation['sampling_intervals_distribution'].items():
        if key not in y3:
            y3[key] = []
        y3[key].append(value)
    for key, value in generation['precisions_distribution'].items():
        if key not in y4:
            y4[key] = []
        y4[key].append(value)

_, axs = plt.subplots(2, 2)

axs[0, 0].stackplot(x, y1.values(), labels=y1.keys(), alpha=0.8)
axs[0, 0].set(
    title='Period',
    xlabel='Generation',
    ylabel='Number of walking strategies',
    xticks=[1] + list(range(100, 301, 100))
)
axs[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))

axs[0, 1].stackplot(x, y2.values(), labels=y2.keys(), alpha=0.8)
axs[0, 1].set(
    title='Type',
    xlabel='Generation',
    ylabel='Number of muscles',
    xticks=[1] + list(range(100, 301, 100))
)
axs[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))

axs[1, 0].stackplot(x, y3.values(), labels=y3.keys(), alpha=0.8)
axs[1, 0].set(
    title='Sampling interval',
    xlabel='Generation',
    ylabel='Number of muscles',
    xticks=[1] + list(range(100, 301, 100))
)
axs[1, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))

axs[1, 1].stackplot(x, y4.values(), labels=y4.keys(), alpha=0.8)
axs[1, 1].set(
    title='Precision',
    xlabel='Generation',
    ylabel='Number of muscles',
    xticks=[1] + list(range(100, 301, 100))
)
axs[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()