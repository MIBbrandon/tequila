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
        super(TFLayer, self).__init__()
        self.input_vars = input_vars
        self.inputs = None  # Empty for now, but when called, it will hold a tensor with the values of the inputs
        self.units = 1  # TODO: condition this to the number of parameters

        self.objective = objective
        if isinstance(objective, tuple) or isinstance(objective, list) or isinstance(objective, Objective):
            objective = vectorize(list_assignment(objective))
            self.objective = objective

        self.comped_objective, self.compile_args, self.weight_vars, self.w_grads, self.i_grads, self.first, \
            self.second = preamble(objective, compile_args, input_vars)

        # VARIABLES

        # If there are inputs, prepare the input as a trainable variable
        if input_vars is not None:
            initializer = tf.constant_initializer(np.random.uniform(low=0., high=2 * np.pi, size=len(input_vars)))
            self.input_variable = self.add_weight(name="input_tensor",
                                                  shape=(len(input_vars)),
                                                  initializer=initializer,
                                                  trainable=True)

        # If there are weight variables, prepare the params as a trainable variable
        if list(self.weight_vars):
            # Initialize the variable tensor that will hold the weights/parameters/angles
            initializer = tf.constant_initializer(np.random.uniform(low=0., high=2 * np.pi, size=len(self.weight_vars)))
            self.params = self.add_weight(name="params", shape=(len(self.weight_vars)), dtype="float32",
                                          initializer=initializer, trainable=True)

        # If the user specified initial values for these parameters, use them
        if compile_args is not None and compile_args["initial_values"] is not None:
            self.params.assign([compile_args["initial_values"][val] for val in compile_args["initial_values"]])

        self._input_len = 0
        if input_vars is not None:
            self._input_len = len(input_vars)
        self.samples = None
        if self.compile_args is not None:
            self.samples = self.compile_args["samples"]

    def __call__(self, input_tensor: tf.Tensor = None) -> tf.Tensor:
        """
        Calls the Objective on a TF tensor object and returns the results.

        There are three cases which we could have:
            1) We have just input variables
            2) We have just parameter variables
            3) We have both input and parameter variables

        We must determine which situation we are in and execute the corresponding _do() function to also get the
        correct gradients.

        Parameters
        ----------
        input_tensor: TF.Tensor, optional:
            a TF tensor. Should have dimensions (any,self._input_len)

        Returns
        -------
        tf.Tensor:
            a TF tensor, the result of calling the underlying objective on the data input.
        """
        if self.input_vars is not None and self.weight_vars:  # If there are both inputs and params
            try:
                self.input_variable.assign(input_tensor)
            except:
                # User forgot to insert input_tensor
                raise TequilaMLException("Please insert the input tensor")

            # Forward pass with both inputs and parameters
            out = self._do(self.input_variable, self.get_params())
        elif self.input_vars is not None:
            try:
                self.input_variable.assign(input_tensor)
            except:
                # User forgot to insert input_tensor
                raise TequilaMLException("Please insert the input tensor")

            # Forward pass with just input
            out = self._do_just_input(self.input_variable)
        else:
            # Only option left is that there are only parameters
            out = self._do_just_params(self.get_params())
        return out

# We need to make a _do() function for where there is an input and another one for when there isn't. This is because the
# decorator @tf.custom_gradient counts the number of required inputs and forces that same number of gradients to appear


    @tf.custom_gradient
    def _do_just_input(self, input_tensor: tf.Tensor):
        """
        Forward pass with just the inputs

        Parameters
        ----------
        input_tensor

        Returns
        -------

        """
        if len(input_tensor.numpy().tolist()) != self._input_len:
            raise TequilaMLException('Received input of len {} when Objective takes {} inputs.'.format(len(input_tensor.numpy()), self._input_len))
        else:
            input_tensor = tf.stack(input_tensor)

        def grad(upstream):
            d_wrt_input = self.get_grads_values(only="inputs")
            return upstream * d_wrt_input

        return self.realForward(inputs=input_tensor, angles=None), grad

    @tf.custom_gradient
    def _do_just_params(self, params_tensor: tf.Tensor):
        """
        Forward pass with just the parameters

        Parameters
        ----------
        params_tensor

        Returns
        -------

        """

        # TODO: check if we need to raise an exception here in relation to the length of the tensor for the parameters
        params_tensor = tf.stack(params_tensor)

        def grad(upstream):
            d_wrt_param = self.get_grads_values(only="params")
            return upstream * d_wrt_param

        return self.realForward(inputs=None, angles=params_tensor), grad

    @tf.custom_gradient
    def _do(self, input_tensor: tf.Tensor, params_tensor: tf.Tensor):
        """
        Forward pass with both input and parameter variables

        Parameters
        ----------
        input_tensor: TF.Tensor, optional:
            Input tensor to involve

        Returns
        -------
        tf.Tensor:
            Result of the forward pass
        """
        # TODO: check if we need to raise an exception here in relation to the length of the tensor for the parameters
        params_tensor = tf.stack(params_tensor)

        if len(input_tensor.numpy().tolist()) != self._input_len:
            raise TequilaMLException('Received input of len {} when Objective takes {} inputs.'.format(len(input_tensor.numpy()), self._input_len))
        else:
            input_tensor = tf.stack(input_tensor)

        def grad(upstream):
            input_gradient_values, parameter_gradient_values = self.get_grads_values()
            # Convert to tensor
            in_Tensor = tf.convert_to_tensor(input_gradient_values, dtype=tf.float32)
            par_Tensor = tf.convert_to_tensor(parameter_gradient_values, dtype=tf.float32)

            # Multiply with the upstream
            in_Upstream = tf.dtypes.cast(upstream, tf.float32) * in_Tensor
            par_Upstream = tf.dtypes.cast(upstream, tf.float32) * par_Tensor

            # Transpose and sum
            return tf.reduce_sum(tf.transpose(in_Upstream), axis=0), tf.reduce_sum(tf.transpose(par_Upstream), axis=0)

        return self.realForward(inputs=input_tensor, angles=params_tensor), grad  # Just get the result, not the grad method

    def realForward(self, inputs: Union[tf.Tensor, None], angles: Union[tf.Tensor, None]) -> tf.Tensor:
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
        list_angles = self.get_params_list()
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

        # We must accumulate the gradients (which I imagine means to add them)
        try:
            if inputs_grads_values:
                # We first turn it into a numpy array, then transpose it
                inputs_grads_values = np.array(inputs_grads_values)

            if param_grads_values:
                # We first turn it into a numpy array, then transpose it
                param_grads_values = np.array(param_grads_values)
        except:
            raise TequilaMLException("Error trying to reshape grads_values")

        # Determine what to return
        # Put in a list since optimizers.apply_gradients() requires so
        if get_input_grads and get_param_grads:
            return inputs_grads_values, param_grads_values
        elif get_input_grads and not get_param_grads:
            return inputs_grads_values
        elif not get_input_grads and get_param_grads:
            return param_grads_values

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

    def get_params(self):
        return self.params

    def get_params_list(self):
        if self.get_params() is not None:
            return self.get_params().numpy().tolist()
        return []

    def get_inputs(self):
        return self.input_variable

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
