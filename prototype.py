from osim.env import L2M2019Env
import numpy as np
import copy
import pickle
import random
from walking_strategy import WalkingStrategy

from concurrent.futures import ProcessPoolExecutor

#increase crossover points dynamically
#increase mutation probabvilitydynamixcally
#сдлееать разлдтчте по фазам просто

def crossover(walking_strategy_1, walking_strategy_2):
    switch_index = random.randint(0, 220)

    fourier_1 = [
        v
        for muscle in walking_strategy_1.muscle_activations_fourier_coefficients
        for v in muscle
    ]

    fourier_2 = [
        v
        for muscle in walking_strategy_2.muscle_activations_fourier_coefficients
        for v in muscle
    ]

    new_muscle_activations_fourier_coefficients = np.array_split(np.concatenate((fourier_1[:switch_index], fourier_2[switch_index:])), 22)

    return WalkingStrategy(400, new_muscle_activations_fourier_coefficients)

class WalkingStrategyPopulation:
    def __init__(self, **kwargs):
        walking_strategies = kwargs.get('walking_strategies')
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        size = kwargs.get('size')
        if size is not None:
            self.walking_strategies = [WalkingStrategy(400) for _ in range(size)]
            return

        raise Exception('Wrong arguments')

    @staticmethod
    def get_fitness_map(fitness_values):
        fitmap = []
        total = 0
        for f in fitness_values:
            total = total + f
            fitmap.append(total)
        return fitmap

    def select_parent(self, fitmap):
        r = np.random.rand()  # 0-1
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return self.walking_strategies[i]

        return self.walking_strategies[len(fitmap) - 1]


iterations = 5000
sim_steps_per_iteration = 1000

population = WalkingStrategyPopulation(size=10)

envs = [L2M2019Env(visualize=False, difficulty=0) for _ in range(len(population.walking_strategies))]

def evaluate(i, walking_strategy):
    global envs
    envs[i].reset()
    total_reward = 0
    for sim_step in range(sim_steps_per_iteration):
        observation, reward, done, info = envs[i].step(walking_strategy.calculate_muscle_activations(sim_step))
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == "__main__":

    def mutate(walking_strategy, mut_coef):
        new_muscle_activations_fourier_coefficients = copy.deepcopy(walking_strategy.muscle_activations_fourier_coefficients)

        for i in range(10):
            for j in range(new_muscle_activations_fourier_coefficients[i].shape[0]):
                if random.random() < 0.05:
                    new_muscle_activations_fourier_coefficients[i][j] += mut_coef * np.random.randn()

        return WalkingStrategy(400, new_muscle_activations_fourier_coefficients)

    executor = ProcessPoolExecutor(max_workers=len(population.walking_strategies))

    mutation_coefficient = 0.1
    total_best_fitness_value = 0
    current_best_fitness_value = 0
    iterations_with_fitness_improvement = 0
    iterations_without_fitness_improvement = 0

    for iteration in range(iterations):
        print(f'Last fitness: {current_best_fitness_value}, Best fitness: {total_best_fitness_value}')
        print(f'Iteration: {iteration + 1}/{iterations}')

        # eval population

        futures = [executor.submit(evaluate, i, walking_strategy) for i, walking_strategy in enumerate(population.walking_strategies)]
        fitness_values = np.array([future.result() for future in futures])

        current_best_fitness_value = fitness_values.max()
        if current_best_fitness_value > total_best_fitness_value:
            total_best_fitness_value = current_best_fitness_value
            iterations_without_fitness_improvement = 0
            iterations_with_fitness_improvement += 1
        else:
            iterations_with_fitness_improvement = 0
            iterations_without_fitness_improvement += 1
        if iterations_without_fitness_improvement > 10:
            mutation_coefficient += 0.1
        elif iterations_with_fitness_improvement > 10:
            mutation_coefficient -= 0.1
        if mutation_coefficient > 2:
            mutation_coefficient = 2
        if mutation_coefficient < 0.1:
            mutation_coefficient = 0.1


        # futures = [evaluate(i, walking_strategy) for i, walking_strategy in enumerate(population.walking_strategies)]
        # fitness_values = np.array(futures)


        # give a birth to a new population
        fit_map = population.get_fitness_map(fitness_values)
        new_walking_strategies = []
        for _ in range(len(population.walking_strategies)):
            parent1 = population.select_parent(fit_map)
            parent2 = population.select_parent(fit_map)
            new_walking_strategy = crossover(parent1, parent2)

            new_walking_strategy = mutate(new_walking_strategy, mutation_coefficient)

            new_walking_strategy.normalize()

            new_walking_strategies.append(new_walking_strategy)

        # preserve elites
        max_fits = -np.partition(-fitness_values, 2)[:2]
        elites_saved = 0
        for i, walking_strategy in enumerate(population.walking_strategies):
            if fitness_values[i] in max_fits:
                new_walking_strategies[elites_saved] = walking_strategy
                elites_saved += 1

            if elites_saved == 2:
                break

        population = WalkingStrategyPopulation(walking_strategies=new_walking_strategies)


    executor.shutdown()

    # save the best
    with open('best', 'wb') as file:
        pickle.dump(population.walking_strategies[0], file)