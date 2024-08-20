# this code is partially based on the code from https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py

import opensim
import os
import numpy as np


class OsimModel:
    """
    Represents OpemSim environment and model using OpenSim API.
    """

    stepsize = 0.01

    model = None
    state = None
    brain = None
    istep = 0

    def __init__(self, mode, visualize, integrator_accuracy=0.001):
        """
        Prepares OpenSim environment and model for simulation.
        """

        # read 2D or 3D model file
        if mode == '3D':
            self.model = opensim.Model(os.path.join(os.path.dirname(__file__), '../models/3d.osim'))
        elif mode == '2D':
            self.model = opensim.Model(os.path.join(os.path.dirname(__file__), '../models/2d.osim'))
        else:
            raise ValueError('Invalid mode')

        # init OpenSIm env
        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()

        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(self.muscleSet.get(j).getName(), func)

        self.model.addController(self.brain)
        self.model.initSystem()

        self.integrator_accuracy = integrator_accuracy

    def actuate(self, action):
        """
        Updates the environment state by muscle tensions.
        """
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        action = np.clip(np.array(action), 0.0, 1.0)

        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    def compute_state_desc(self):
        """
        Receives from OpenSim env the current state of the model.
        """
        self.model.realizeAcceleration(self.state)

        res = {
            "body_pos": {},
            "forces": {},
            "muscles": {}
        }

        ## Bodies
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]

        ## Forces
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {"activation": muscle.getActivation(self.state)}

        return res

    def reset_manager(self):
        """
        Resets the OpenSim environment and model.
        """
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.istep = int(self.state.getTime() / self.stepsize)
        self.reset_manager()

    def integrate(self):
        """
        Moves the OpenSim environment to the next simulation step.
        """
        self.istep = self.istep + 1
        self.state = self.manager.integrate(self.stepsize * self.istep)