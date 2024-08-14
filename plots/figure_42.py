from walking_strategy.walking_strategy import WalkingStrategy
import matplotlib.pyplot as plt

walking_strategy = WalkingStrategy( mode='2D')
x1 = [i for i in range(walking_strategy.period)]
y1 = [walking_strategy.muscles[0].get_activation(i) for i in range(walking_strategy.period)]

for _ in range(1):
    walking_strategy = walking_strategy.mutate(0.8, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3)
x2 = [i for i in range(walking_strategy.period)]
y2 = [walking_strategy.muscles[0].get_activation(i) for i in range(walking_strategy.period)]

for _ in range(9):
    walking_strategy = walking_strategy.mutate(0.8, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3)
x3 = [i for i in range(walking_strategy.period)]
y3 = [walking_strategy.muscles[0].get_activation(i) for i in range(walking_strategy.period)]

for _ in range(90):
    walking_strategy = walking_strategy.mutate(0.8, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3)
x4 = [i for i in range(walking_strategy.period)]
y4 = [walking_strategy.muscles[0].get_activation(i) for i in range(walking_strategy.period)]

plt.plot(x1, y1, label="original")
plt.plot(x2, y2, label="mutated 1 time")
plt.plot(x3, y3, label="mutated 10 times")
plt.plot(x4, y4, label="mutated 100 times")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()