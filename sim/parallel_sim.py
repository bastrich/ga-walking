from sim.sim import Sim
from concurrent.futures import ProcessPoolExecutor
from queue import Queue


class ParallelSim():
    def __init__(self, mode, parallelization):
        self.sims = Queue(maxsize=parallelization)
        for _ in range(parallelization):
            self.sims.put(Sim(mode, False))

        self.sims_executor = ProcessPoolExecutor(max_workers=parallelization)

    def run(self, walking_strategies, number_of_steps=1000):
        futures = [self.sims_executor.submit(self.evaluate, walking_strategy, number_of_steps) for walking_strategy in walking_strategies]
        return [future.result() for future in futures]

    def evaluate(self, walking_strategy, number_of_steps):
        sim = self.sims.get()
        fitness, steps = sim.run(walking_strategy, number_of_steps)
        self.sims.put(sim)

        return walking_strategy, fitness, steps

    def __del__(self):
        self.sims_executor.shutdown()
