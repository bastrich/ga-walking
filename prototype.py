from my_osim import L2M2019Env
import numpy as np
import copy
import pickle
import random
from walking_strategy import WalkingStrategy
from walking_strategy_population import WalkingStrategyPopulation

from concurrent.futures import ProcessPoolExecutor

import time

period = 200

def crossover(walking_strategy_1, walking_strategy_2):
    # switch_indexes = np.random.randint(1, 175, np.random.randint(1, 5))
    switch_indexes = np.random.randint(1, 439, 1)
    switch = True
    new_muscle_activations = []

    for i in range(440):
        if i in switch_indexes:
            switch = not switch
        if switch is True:
            new_muscle_activations.append(walking_strategy_1.muscle_activations[i // 40][i % 40])
        else:
            new_muscle_activations.append(walking_strategy_2.muscle_activations[i // 40][i % 40])

    new_muscle_activations = np.array_split(new_muscle_activations, 11)

    return WalkingStrategy(period, new_muscle_activations)


iterations = 10000
sim_steps_per_iteration = 1000

# population = WalkingStrategyPopulation(period, size=30)
with open('population', 'rb') as file:
    population = pickle.load(file)

envs = [L2M2019Env(visualize=False, difficulty=0) for _ in range(len(population.walking_strategies))]

def evaluate(i, walking_strategy):
    global envs
    envs[i].reset()
    total_reward = 0
    for sim_step in range(sim_steps_per_iteration):
        observation, reward, done, info = envs[i].step(walking_strategy.get_muscle_activations(sim_step))
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == "__main__":

    def mutate(walking_strategy, mutation_rate):
        new_muscle_activations = copy.deepcopy(walking_strategy.muscle_activations)

        for i in range(11):
            for j in range(40):
                # if random.random() < shrink_growth_rate:
                #     new_muscle_activations_fourier_coefficients[i][j] = random.choice([-100, 0, 100])
                if np.random.uniform() < mutation_rate:
                    scale = new_muscle_activations[i][j]
                    if scale <= 0.03:
                        scale = 0.03

                    new_value = np.random.normal(loc=new_muscle_activations[i][j], scale=scale)

                    if new_muscle_activations[i][j] <= 0 and new_value <= 0:
                        new_value = np.abs(new_value)
                    elif new_muscle_activations[i][j] >= 1 and new_value >= 1:
                        new_value = 2 - new_value

                    if new_value > 1:
                        new_value = 1
                    elif new_value < 0:
                        new_value = 0

                    new_muscle_activations[i][j] = new_value


        return WalkingStrategy(period, new_muscle_activations)

    executor = ProcessPoolExecutor(max_workers=len(population.walking_strategies))

    # shrink_growth_rate = 0.01
    mutation_rate = 0.5
    # mutation_coefficient = 0.01

    total_best_fitness_value = -1000
    current_best_fitness_value = -1000

    iterations_with_fitness_improvement = 0
    iterations_without_fitness_improvement = 0

    start_time = time.time()

    for iteration in range(iterations):
        print(f'Last fitness: {current_best_fitness_value}, Best fitness: {total_best_fitness_value}')
        print(f'Iteration: {iteration + 1}/{iterations}')

        # eval population

        futures = [executor.submit(evaluate, i, walking_strategy) for i, walking_strategy in enumerate(population.walking_strategies)]
        fitness_values = np.array([future.result() for future in futures])

        current_best_fitness_value = fitness_values.max()
        if current_best_fitness_value > total_best_fitness_value + 0.01:
            iterations_without_fitness_improvement = 0
            iterations_with_fitness_improvement += 1
        elif current_best_fitness_value <= total_best_fitness_value:
            iterations_with_fitness_improvement = 0
            iterations_without_fitness_improvement += 1
        if current_best_fitness_value > total_best_fitness_value:
            total_best_fitness_value = current_best_fitness_value

        if iterations_without_fitness_improvement > 30:
            print('30 generations without improvement, increasing mutation rate')
            # shrink_growth_rate += 0.01
            mutation_rate += 0.1
            # mutation_coefficient += 0.01
            iterations_without_fitness_improvement = 0
        elif iterations_with_fitness_improvement > 0:
            print('1 generation with improvement, decreasing mutation rate')
            # shrink_growth_rate -= 0.01
            mutation_rate -= 0.1
            # mutation_coefficient -= 0.01


        # give a birth to a new population
        new_walking_strategies = []

        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values + np.abs(min_fitness)

        # preserve elites
        max_fits = -np.partition(-fitness_values, 3)[:3]
        elites_saved = 0
        for i, walking_strategy in enumerate(population.walking_strategies):
            if fitness_values[i] in max_fits:
                new_walking_strategies.append(walking_strategy)
                elites_saved += 1

            if elites_saved == 3:
                break


        # shrink_growth_rate = np.clip(shrink_growth_rate, 0.01, 0.1)
        mutation_rate = np.clip(mutation_rate, 0.1, 0.9)
        # mutation_coefficient = np.clip(mutation_coefficient, 0.1, 5)

        fit_map = population.get_fitness_map(fitness_values)

        for _ in range(len(population.walking_strategies) - 3):
            parent1, parent2 = population.select_parents(fit_map)
            new_walking_strategy = crossover(parent1, parent2)

            new_walking_strategy = mutate(new_walking_strategy, mutation_rate)

            new_walking_strategies.append(new_walking_strategy)



        population = WalkingStrategyPopulation(period, walking_strategies=new_walking_strategies)
        # save current population
        with open(f'population', 'wb') as file:
            pickle.dump(population, file)

    end_time = time.time()

    print(f'Execution time: {(end_time - start_time) / 60} minutes')

    executor.shutdown()

