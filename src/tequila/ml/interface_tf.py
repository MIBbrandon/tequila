from typing import List

from typing import Union, Dict, Any

from .utils_ml import preamble, TequilaMLException
from tequila.objective import Objective, VectorObjective, Variable, vectorize
from tequila.tools import list_assignment
from tequila.simulators.simulator_api import simulate
import numpy as np

import tensorflow as tf

class TFLayer(tf.keras.layers.Layer):
    def __init__(self, objective: Union[Objective, VectorObjective], compile_args: Dict[str, Any] = None,
                 input_vars: Dict[str, Any] = None, **kwargs):
        """
        Tensorflow layer that compiles the Objective (or VectorObjective) with the given compile arguments and/or
        input variables if there are any when initialized. When called, it will forward the input variables into the
        compiled objective (if there are any inputs needed) and will return the output. The gradient values can also
        be returned.

        Parameters
        ----------
        objective: Objective or VectorObjective to compile and run.
        compile_args: dict of all the necessary information to compile the objective
        input_vars: List of variables that will be inputs
        """
        super(TFLayer).__init__(**kwargs)
        self.input_vars = input_vars
        self.inputs = None  # Empty for now, but when called, it will hold a tensor with the values of the inputs

        self.objective = objective
        if isinstance(objective, tuple) or isinstance(objective, list) or isinstance(objective, Objective):
            objective = vectorize(list_assignment(objective))
            self.objective = objective

        # If no initial values were given, we start with random values
        if compile_args is None or compile_args["initial_values"] is None:
            compile_args = {}
            compile_args["initial_values"] = {}
            vars = sorted(self.objective.extract_variables())
            for var_name in vars:
                # We assign initial values to variables which are not input variables
                if input_vars is None or var_name not in input_vars:
                    compile_args["initial_values"][var_name] = np.random.uniform(low=0, high=2 * np.pi)

            if compile_args["initial_values"]:
                self.angles = tf.Variable([compile_args["initial_values"][val] for val in compile_args["initial_values"]])
            else:
                self.angles = None
        else:
            self.angles = tf.Variable([compile_args["initial_values"][val] for val in compile_args["initial_values"]])

        self.comped_objective, self.compile_args, self.weight_vars, self.w_grads, self.i_grads, self.first, \
            self.second = preamble(objective, compile_args, input_vars)

        self._input_len = 0
        if input_vars is not None:
            self._input_len = len(input_vars)
        self.samples = None
        if self.compile_args is not None:
            self.samples = self.compile_args["samples"]

    def __call__(self, input_tensor: tf.Tensor = None) -> tf.Tensor:
        """
        Calls the Objective on a TF tensor object and returns the results.

        Parameters
        ----------
        input_tensor: TF.Tensor, optional:
            a TF tensor. Should have dimensions (any,self._input_len)

        Returns
        -------
        tf.Tensor:
            a TF tensor, the result of calling the underlying objective on the data input.
        """
        self.inputs = input_tensor
        if input_tensor is not None:
            if len(input_tensor.shape) == 1:
                out = self._do(input_tensor)
            else:
                out = tf.stack([self._do(y) for y in input_tensor])
        else:
            out = self._do(None)
        return out

    def _do(self, input_tensor: tf.Tensor = None) -> tf.Tensor:
        """
        If there is input to involve in this forward pass, involve it. Otherwise, do a normal forward pass.

        Parameters
        ----------
        input_tensor: TF.Tensor, optional:
            Input tensor to involve

        Returns
        -------
        tf.Tensor:
            Result of the forward pass
        """
        try:
            # Try to get self.angles
            listed = self.get_angles()
            if listed is None:
                raise Exception  # Just to trigger the except
            # TODO: focus on this case
            f = tf.stack(listed)
        except:
            f = None
        if input_tensor is not None:
            if len(input_tensor.numpy().tolist()) != self._input_len:
                raise TequilaMLException('Received input of len {} when Objective takes {} inputs.'.format(len(input_tensor), self._input_len))
            else:
                input_tensor = tf.stack(input_tensor)
        return self.realForward(inputs=input_tensor, angles=f)

    def realForward(self, inputs: tf.Tensor, angles: tf.Tensor) -> tf.Tensor:
        """
        This is where we really execute the forward pass.

        Parameters
        ----------
        inputs
        angles

        Returns
        -------

        """
        def tensor_fix(inputs_tensor: Union[tf.Tensor, None], params_tensor: Union[tf.Tensor, None],
                       first: Dict[int, Variable], second: Dict[int, Variable]):
            """
            Prepare a dict with the right information about the involved variables (whether input or parameter) and
            their corresponding values.

            Note: if "inputs_tensor" and "angles_tensor" are None or "first" and "second" are empty dicts, something
            went wrong, since the objective should have either inputs or parameters to tweak.

            Parameters
            ----------
            inputs_tensor
                Tensor holding the values of the inputs
            params_tensor
                Tensor holding the values of the parameters
            first
                Dict mapping numbers to input variable names
            second
                Dict mapping numbers to parameter variable names

            Returns
            -------
            back
                Dict mapping all variable names to values
            """
            back = {}
            if inputs_tensor is not None:
                for i, val in enumerate(inputs_tensor):
                    back[first[i]] = val.numpy()
            if params_tensor is not None:
                for i, val in enumerate(params_tensor):
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

    def get_grads_values(self, only: str = None):
        """
        Gets the values of the gradients with respect to the inputs and the parameters

        Returns
        -------
        grad_values
            If "only" is None, a tuple of two elements, the first one being a list of gradients to apply to the input
            variables, and the second element being a list of gradients to apply to the parameter variables.

            If only == inputs, just the list of gradient values w.r.t. the input variables.
            If only == params, just the list of gradient values w.r.t. the parameter variables.
        """
        get_input_grads = True
        get_param_grads = True

        # Determine which gradients to calculate
        if only is not None:
            if only == "inputs":
                get_input_grads = True
                get_param_grads = False
            elif only == "params":
                get_input_grads = False
                get_param_grads = True
            else:
                raise TequilaMLException("Valid values for \"only\" are \"inputs\" and \"params\".")


        # Get the current values of the inputs and parameters in a dict called "variables"
        # Inputs
        variables = {}
        list_inputs = self.get_inputs_list()
        in_vars = None
        if list_inputs:
            in_vars = sorted(self.input_vars)
            for i, in_var_name in enumerate(in_vars):
                variables[in_var_name] = list_inputs[i]

        # Parameters
        list_angles = self.get_angles_list()
        param_vars = None
        if list_angles:
            param_vars = sorted(self.weight_vars)
            for p, param_name in enumerate(param_vars):
                variables[param_name] = list_angles[p]

        # Get the gradient values with respect to the inputs
        inputs_grads_values = []
        if get_input_grads and in_vars:
            for in_var in sorted(in_vars):
                self.get_grad_values(inputs_grads_values, in_var, variables, self.i_grads)

        # Get the gradient values with respect to the parameters
        param_grads_values = []
        if get_param_grads and param_vars:
            for param_var in sorted(param_vars):  # Iterate through the names of the parameters
                self.get_grad_values(param_grads_values, param_var, variables, self.w_grads)

        # Given the gradients per expectation value, we want to use the most informative, which
        # has the highest absolute value
        try:
            if inputs_grads_values:
                for in_val in range(len(inputs_grads_values)):
                    # Get max absolute value while preserving the sign
                    inputs_grads_values[in_val] = max(inputs_grads_values[in_val], key=abs)
            if param_grads_values:
                for param in range(len(param_grads_values)):
                    # Get max absolute value while preserving the sign
                    param_grads_values[param] = max(param_grads_values[param], key=abs)
        except:
            raise TequilaMLException("Error trying to reshape grads_values")

        # Determine what to return
        # Put in a list since optimizers.apply_gradients() requires so
        if get_input_grads and get_param_grads:
            return [inputs_grads_values], [param_grads_values]
        elif get_input_grads and not get_param_grads:
            return [inputs_grads_values]
        elif not get_input_grads and get_param_grads:
            return [param_grads_values]

    def get_grad_values(self, grads_values, var, variables, objectives_grad):
        """
        Inserts into "grads_values" the gradient values per objective in objectives_grad[var], where var is the name
        of the variable.

        Parameters
        ----------
        grads_values
            List in which we insert the gradient values (No returns)
        var
            Variable over which we are calculating the gradient values
        variables
            Dict mapping all variables to their current values
        objectives_grad
            List of ExpectationValueImpls that will be simulated to calculate the gradient value of a given variable
        """
        var_results = []
        grads_wrt_var = objectives_grad[var]
        if not isinstance(grads_wrt_var, List):
            grads_wrt_var = [grads_wrt_var]
        for obj in grads_wrt_var:
            # TODO: decide whether to directly simulate or to go through the preamble() to the call(). For now, sim.
            var_results.append(simulate(objective=obj, variables=variables,
                                        backend=self.compile_args["backend"],
                                        samples=self.samples))
        grads_values.append(var_results)

    def get_angles(self):
        return self.angles

    def get_angles_list(self):
        if self.get_angles() is not None:
            return self.get_angles().numpy().tolist()
        return []

    def get_inputs(self):
        return self.inputs

    def get_inputs_list(self):
        if self.get_inputs() is not None:
            return self.get_inputs().numpy().tolist()
        return []

    def __repr__(self) -> str:
        """
        Returns
        -------
        str:
            Information used by print(TFLayer).
        """
        string = 'Tequila TFLayer. Represents: \n'
        string += '{} \n'.format(str(self.objective))
        string += 'Current Weights: {}'.format(list(self.weight_vars))
        return string
