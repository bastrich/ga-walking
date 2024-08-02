from sim.sim import Sim
from concurrent.futures import ProcessPoolExecutor
from queue import Queue


class PopulationEvaluator:
    def __init__(self, mode, parallelization):
        self.sims = Queue(maxsize=parallelization)
        for _ in range(parallelization):
            self.sims.put(Sim(mode, False))

        self.sims_executor = ProcessPoolExecutor(max_workers=parallelization)

    def breed_new_population(self, simulation_results, mutation_parameters):
        populations_executor = ProcessPoolExecutor(max_workers=30)
        populations_executor.shutdown()

        # give a birth to a new population
        new_walking_strategies = []

        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values + np.abs(min_fitness)

        print('Selecting elites...')

        # preserve elites
        number_to_preserve = 20
        preserved_walking_strategies = [walking_strategy for walking_strategy in
                                        heapq.nlargest(number_to_preserve, walking_strategies,
                                                       key=lambda walking_strategy: walking_strategy.evaluated_fitness)]
        # np.random.shuffle(preserved_walking_strategies)
        new_walking_strategies += preserved_walking_strategies

        # shrink_growth_rate = np.clip(shrink_growth_rate, 0.01, 0.1)
        mutation_rate = np.clip(mutation_rate, 0.05, 1)
        # mutation_amount = np.clip(mutation_amount, 0.1, 3)
        mutation_amount = np.clip(mutation_amount, 0.05, 1)
        # mutation_coefficient = np.clip(mutation_coefficient, 0.1, 5)

        # preserve elites with mutation
        # preserved_walking_strategies_with_mutation = [walking_strategy.mutate(mutation_rate, mutation_amount) for walking_strategy in preserved_walking_strategies]
        # new_walking_strategies += preserved_walking_strategies_with_mutation

        fit_map = fitness_values / np.sum(fitness_values)

        print('Creating new population...')

        new_walking_strategies_futures = [
            populations_executor.submit(give_birth_to_new_walking_strategy, population.walking_strategies, fit_map,
                                        mutation_rate, mutation_amount) for _ in
            range(len(population.walking_strategies) - number_to_preserve)]
        new_walking_strategies += [future.result() for future in new_walking_strategies_futures]

        population = WalkingStrategyPopulation(walking_strategies=new_walking_strategies)


        futures = [self.sims_executor.submit(self.evaluate, walking_strategy, number_of_steps) for walking_strategy in walking_strategies]
        return [future.result() for future in futures]

    def evaluate(self, walking_strategy, number_of_steps):
        sim = self.sims.get()
        fitness, steps = sim.run(walking_strategy, number_of_steps)
        self.sims.put(sim)

        return {
            'walking_strategy': walking_strategy,
            'fitness': fitness,
            'steps': steps
        }

    def give_birth_to_new_walking_strategy(walking_strategies, fit_map, mutation_rate, mutation_amount):
        parent1, parent2 = np.random.choice(walking_strategies, size=2, p=fit_map)
        new_walking_strategy = parent1.crossover(parent2)
        new_walking_strategy = new_walking_strategy.mutate(mutation_rate, mutation_amount)
        return new_walking_strategy

    def __del__(self):
        self.sims_executor.shutdown()
