from sim.sim import Sim
from multiprocessing import Process, Queue


class ParallelSim:
    """
    Runs parallel simulations using multiprocessing.

    Attributes:
        mode (str): 2D or 3D.
        parallelization (int): The number of parallel processes.
        integrator_accuracy (float): OpenSim accuracy coniguration.
        input_queue (Queue): A queue for sending tasks to worker processes.
        output_queue (Queue): A queue for receiving results from worker processes.
        processes (list): A list of worker processes.
    """

    def __init__(self, mode, parallelization, integrator_accuracy):
        """
        Initializes the parallel simulation and starts processes.
        """
        self.mode = mode
        self.parallelization = parallelization
        self.integrator_accuracy = integrator_accuracy
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.processes = []
        for _ in range(self.parallelization):
            process = Process(target=self.simulation_worker, args=(self.mode, self.integrator_accuracy, self.input_queue, self.output_queue))
            self.processes.append(process)
            process.start()

    @staticmethod
    def simulation_worker(mode, integrator_accuracy, input_queue, output_queue):
        """
        Performs the actual simulation work in a separate process.
        """
        try:
            sim = Sim(mode, False, integrator_accuracy)
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
        """
        Runs the simulation for each walking strategy in parallel and collects the results.
        """
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
        """
        Gracefully shuts down all parallel processes.
        """
        for _ in range(self.parallelization):
            self.input_queue.put(None)
        for process in self.processes:
            process.join()
