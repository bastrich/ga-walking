from osim.env import L2M2019Env
import numpy as np
import copy
import pickle

# TODO: we should exploit the Fourier property for which higher harmonics weights tend to decays as 1/x^n for smooth and continous functions
# working with pickle
# fileObject = open(file_Name, 'r')
# w_best = pickle.load(fileObject)
# pickle.dump(w_best,fileObject)


class WalkingStrategy:
    def __init__(self, period):
        self.muscle_activations_fourier_coefficients = [WalkingStrategy.generate_single_muscle_activation() for _ in range(22)]
        self.period = period

    @staticmethod
    def generate_single_muscle_activation():
        result = []

        a0 = np.random.uniform(-10, 10)
        result.append(a0)

        current_sum = a0
        current_negative_sum = a0

        for _ in range(9):
            coefficient = np.random.uniform(-10, min(1 - current_sum, current_negative_sum))

            result.append(coefficient)

            current_sum += coefficient
            current_negative_sum -= coefficient

        return result

    def calculate_muscle_activations(self, time):
        return [self.calculate_fourier_series_sum(coefficients, time) for coefficients in self.muscle_activations_fourier_coefficients]

    def calculate_fourier_series_sum(self, coefficients, time):
        a0 = coefficients[0]
        result = a0

        for i in range(1, len(coefficients), 3):
            an = coefficients[i]
            bn = coefficients[i + 1]
            phase = coefficients[i + 2]
            result += an * np.cos((i // 3 + 1) * 2 * np.pi * time / self.period + phase) + bn * np.cos(
                (i // 3 + 1) * 2 * np.pi * time / self.period + phase)

        return result

class WalkingStrategyPopulation:
    def __init__(self, size):
        self.walking_strategies = [WalkingStrategy() for _ in range(size)]




def evolve(w):
    # This functions evolves randomly w generating a direction, sampling from gaussians distribution.
    # It operates directly on w so it doesn't return anything.

    for i in range(9):
        w[i] += np.random.randn(8) * alpha
    """
    for i in range(9):
        delta=[]
        for j in range(4):
            delta.append(np.random.randn()*0.5/((j+1)**2))
        delta=np.asarray(delta)
        w[i]+=delta
    """

w_try = copy.deepcopy(w)
best_reward = 0.
runs = 20
unev_runs = 0

print("Baseline, run with w_best")
observation = env.reset()
total_reward = 0.0
for i in range(500):
    i *= 0.01
    if i > 2:
        i -= 2
        observation, reward, done, info = env.step(generate_input(w_best, i))
        T = 2
    else:
        # make a step given by the controller and record the state and the reward
        observation, reward, done, info = env.step(generate_input(w_first, i))
    total_reward += reward
    if done:
        break
best_reward = total_reward

# Your reward is
print("Total reward %f" % total_reward)

for run in range(runs):

    T = 4
    # if it doens't get better for more than 10 iterations, increase alpha to allow bigger changes
    # Increase alpha then set unev_runs back to 0
    if unev_runs > 30:
        print("Augmenting alpha")
        alpha += alpha_0
        unev_runs = 0

    unev_runs += 1
    print("Run {}/{}".format(run, runs))
    observation = env.reset()

    # I copy the best performing w and I try to evolve it
    w_try = copy.deepcopy(w_best)
    evolve(w_try)

    total_reward = 0.0
    for i in range(500):
        # make a step given by the controller and record the state and the reward
        i *= 0.01  # Every step is 0.01 s
        if i > 2:
            T = 2
            i -= 2
            observation, reward, done, info = env.step(generate_input(w_try, i))
        else:
            # make a step given by the controller and record the state and the reward

            observation, reward, done, info = env.step(generate_input(w_first, i))
        total_reward += reward
        if done:
            print("done")
            break

    if total_reward > best_reward:
        # If the total reward is the best one, I store w_try as w_best, dump it with pickle and save the reward
        print("Found a better one!")
        unev_runs = 0
        alpha = alpha_0
        w_best = copy.deepcopy(w_try)
        print(w_best)

        fileObject = open(file_Name, 'wb')
        # pickle.dump(w_best,fileObject)
        fileObject.close()
        best_reward = total_reward

    # Your reward is
    print("Total reward %f" % total_reward)

# Final run with video and best weights. The raw_input waits for the user to type something. (if it's afk)
print("Run with best weights")
_ = input("ready? ")
env = L2M2019Env(visualize=True)
observation = env.reset()

T = 4
total_reward = 0.0
for i in range(500):
    # make a step given by the controller and record the state and the reward
    i *= 0.01  # Every step is 0.01 s

    if i > 2:
        i -= 2
        T = 2
        observation, reward, done, info = env.step(generate_input(w_best, i))
    else:
        # make a step given by the controller and record the state and the reward

        observation, reward, done, info = env.step(generate_input(w_first, i))
    total_reward += reward
    if done:
        print("done")
        break

# Your reward is
print("Total reward %f" % total_reward)

print("best weights")
print(w_best)