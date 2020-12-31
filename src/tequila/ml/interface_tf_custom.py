from typing import List

import tequila as tq
from .utils_ml import preamble, TequilaMLException
from tequila.objective import Objective, vectorize, Variable
from tequila.tools import list_assignment
from tequila.circuit.gradient import grad
from tequila.simulators.simulator_api import simulate
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers

class TFLayer(tf.keras.layers.Layer):
    def __init__(self, objective: Objective, compile_args=None, input_vars=None):
        super(TFLayer, self).__init__()
        self.input_vars = input_vars
        self.objective = objective
        if isinstance(objective, tuple) or isinstance(objective, list) or isinstance(objective, Objective):
            objective = vectorize(list_assignment(objective))
            self.objective = objective
        self.comped_objective, self.compile_args, self.weight_vars, self.w_grads, self.i_grads, self.first, \
            self.second = preamble(objective, compile_args, input_vars)
        initial_values = self.compile_args["initial_values"]
        if initial_values is not None:
            self.init_params = initial_values.store
            self.angles = tf.Variable([self.init_params[val] for val in self.init_params])
        """
        Dict with keys being the parameter names, and values being a list of Objectives which, when simulated with 
        certain values, will return the corresponding gradient value.
        """
        self.grads = grad(objective)
        self.samples = compile_args['samples']

    def build(self, input_shape):
        # From the initial weights dict, we create weights
        # for weight_name in self.initial_weight_vars:
        #     self.weight_vars.append(tf.Variable(name=weight_name, initial_value=self.initial_weight_vars[weight_name]))
        self.optimizer = optimizers.SGD(self.get_weights(), lr=.1, momentum=0.9)

    def call(self, toIgnore, x=None):
        """
        Calls the Objective on a torch Tensor object and returns the results.
        Parameters
        ----------
        x: tf.Tensor, optional:
            a tf tensor. Should have dimensions (any,self._input_len)

        Returns
        -------
        torch.Tensor:
            a TF tensor, the result of calling the underlying objective on the data input.
        """
        if x is not None:
            if len(x.shape) == 1:
                out = self._do(x)
            else:
                out = tf.stack([self._do(y) for y in x])
        else:
            # TODO: focus on this case
            out = self._do(None)
        return out

    def _do(self, x):
        """
        If there is something extra to involve in this forward pass, involve it. Otherwise, do a normal forward pass.
        Parameters
        ----------
        x

        Returns
        -------

        """
        listed = self.get_angles()
        try:
            # TODO: focus on this case
            f = tf.stack(listed)
        except:
            f = None
        if x is not None:
            if len(x) != self._input_len:
                raise TequilaMLException('Received input of len {} when Objective takes {} inputs.'.format(len(x),self._input_len))
        return self.realForward(inputs=x, angles=f)

    def realForward(self, inputs, angles):
        def tensor_fix(tensor, angles, first, second):
            """
            take a pytorch tensor and a dict of  int,Variable to create a variable,float dictionary therefrom.
            Parameters
            ----------
            tensor: torch.Tensor:
                a tensor.
            angles: torch.Tensor:
            first: dict:
                dict of int,Variable pairs indicating which position in Tensor corresponds to which variable.
            second: dict:
                dict of int,Variable pairs indicating which position in angles corresponds to which variable.
            Returns
            -------
            dict:
                dict of variable, float pairs. Can be used as call arg by underlying tq objectives
            """
            back = {}
            if tensor is not None:
                for i, val in enumerate(tensor):
                    back[first[i]] = val.numpy()
            if angles is not None:
                for i, val in enumerate(angles):
                    back[second[i]] = val.numpy()
            return back

        call_args = tensor_fix(inputs, angles, self.first, self.second)
        result = self.comped_objective(variables=call_args, samples=self.samples)
        if not isinstance(result, np.ndarray):
            # this happens if the Objective is a scalar since that's usually more convenient for pure quantum stuff.
            result = np.array(result)
        if hasattr(inputs, 'device'):
            if inputs.device == 'cuda':
                r = tf.convert_to_tensor(result).to(inputs.device)
            else:
                r = tf.convert_to_tensor(result)
        else:
            r = tf.convert_to_tensor(result)
        return r

    def get_grads_values(self, x=None):
        """
        Returns the values of the gradients with the current parameters
        """
        # Get the current values of the parameters as a dict
        variables = {}
        list_angles = self.get_angles_list()
        for i, param_name in enumerate(self.init_params):
            variables[param_name] = list_angles[i]

        vars = self.objective.extract_variables()
        grads_values = []
        for var in sorted(vars):  # Iterate through the names of the parameters
            var_results = []
            grads_wrt_var = grad(self.objective, variable=var)
            if not isinstance(grads_wrt_var, List):
                grads_wrt_var = [grads_wrt_var]
            for obj in grads_wrt_var:
                # TODO: decide whether to directly simulate or to go through the preamble() to the call(). For now, sim.
                var_results.append(simulate(objective=obj, variables=variables,
                                            backend=self.compile_args["backend"], samples=self.samples))
            grads_values.append(var_results)

        # Given the gradients per expectation value, we want to use the most informative, which
        # has the highest absolute value
        try:
            for param in range(len(grads_values)):
                # Get max absolute value while preserving the sign
                grads_values[param] = max(grads_values[param], key=abs)
        except:
            raise TequilaMLException("Error trying to reshape grads_values")

        return [grads_values]  # Put in a list since optimizers.apply_gradients() requires so

    def get_angles(self):
        return self.angles

    def get_angles_list(self):
        return self.angles.numpy().tolist()
