from sim.sim import Sim
from multiprocessing import Process, Queue

class ParallelSim:

    def __init__(self, mode, parallelization):
        self.mode = mode
        self.parallelization = parallelization
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.processes = []
        for _ in range(self.parallelization):
            process = Process(target=self.simulation_worker, args=(self.mode, self.input_queue, self.output_queue))
            self.processes.append(process)
            process.start()

    @staticmethod
    def simulation_worker(mode, input_queue, output_queue):
        try:
            sim = Sim(mode, False)
            while True:
                task = input_queue.get()
                if task is None:
                    break

                walking_strategy, number_of_steps = task
                fitness, steps, distance, energy = sim.run(walking_strategy, number_of_steps)

                output_queue.put({
                    'walking_strategy': walking_strategy,
                    'fitness': fitness,
                    'steps': steps,
                    'distance': distance,
                    'energy': energy
                })
        except Exception as e:
            output_queue.put(e)

    def run(self, walking_strategies, number_of_steps=1000):
        for walking_strategy in walking_strategies:
            self.input_queue.put((walking_strategy, number_of_steps))

        simulation_results = []
        for _ in walking_strategies:
            result = self.output_queue.get()
            if isinstance(result, Exception):
                raise result
            simulation_results.append(result)
        return simulation_results

    def shutdown(self):
        for _ in range(self.parallelization):
            self.input_queue.put(None)
        for process in self.processes:
            process.join()