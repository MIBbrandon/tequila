from .utils_ml import preamble, TequilaMLException
from tequila.objective import Objective,vectorize
from tequila.tools import list_assignment
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers

class TFLayer(tf.keras.layers.Layer):
    def __init__(self, objective: Objective, compile_args=None, input_vars=None):
        super(TFLayer, self).__init__()

        if isinstance(objective, tuple) or isinstance(objective, list) or isinstance(objective, Objective):
            objective = vectorize(list_assignment(objective))
        comped_objective, compile_args, weight_vars, w_grads, i_grads, first, \
            second = preamble(objective, compile_args, input_vars)
        self.comped_objective = comped_objective
        # self.compile_args = compile_args
        self.weight_vars = weight_vars
        # self.w_grads = w_grads
        self.i_grads = i_grads
        self.first = first
        self.second = second
        self.samples = compile_args['samples']
        toTensor = [[compile_args['initial_values'][x]] for x in compile_args['initial_values']]
        self.w = tf.Variable(
            toTensor,
            trainable=True,
        )



    def build(self, input_shape):
        # self.kernel = self.add_weight("kernel",
        #                               shape=[int(input_shape[-1]),
        #                                      self.num_outputs])

        self.optimizer = optimizers.SGD(self.get_weights(), lr=.1, momentum=0.9)

    def call(self, inputs, x=None):
        """
        Calls the Objective on a torch Tensor object and returns the results.
        Parameters
        ----------
        x: torch.Tensor, optional:
            a torch tensor. Should have dimensions (any,self._input_len)

        Returns
        -------
        torch.Tensor:
            a PyTorch tensor, the result of calling the underlying objective on the data input.
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
        listed = self.get_weights()
        if listed:
            # TODO: focus on this case
            f = tf.stack(listed)
        else:
            f = None
        if x is not None:
            if len(x) != self._input_len:
                raise TequilaMLException('Received input of len {} when Objective takes {} inputs.'.format(len(x),self._input_len))
        return self.realForward(inputs=x, angles=f)

    def realForward(self, inputs, angles):
        def tensor_fix(tensor: tf.Tensor, angles: tf.Tensor, first: dict, second: dict):
            """
            take a tensorflow tensor and a dict of  int,Variable to create a variable,float dictionary therefrom.
            Parameters
            ----------
            tensor: tf.Tensor:
                a tensor.
            angles: tf.Tensor:
            first: dict:
                dict of int,Variable pairs indicating which position in Tensor corresponds to which variable.
            second: dict:
                dict of int,Variable pairs indicating which position in angles corresponds to which variable.
            Returns
            -------
            dict:
                dict of variable, float pairs. Can be used as call arg by underlying tq objectives
            """

            def flatten(list_of_lists):
                if len(list_of_lists) == 0:
                    return list_of_lists
                if isinstance(list_of_lists[0], list):
                    return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
                return list_of_lists[:1] + flatten(list_of_lists[1:])

            def _check_if_garbage(arg: tf.Tensor):  # Return True if garbage, false if not
                # TODO: apply stricter conditions for arg to be garbage or not
                try:
                    return tf.equal(tf.size(arg), 0)  # Empty tensor is garbage
                except:
                    return True

            back = {}
            if not _check_if_garbage(tensor):
                for i, val in enumerate(tensor):
                    back[first[i]] = val
            if not _check_if_garbage(angles):
                angles = flatten(angles.numpy().tolist())
                for i, val in enumerate(angles):
                    back[second[i]] = val
            return back

        call_args = tensor_fix(inputs, angles, self.first, self.second)
        result = self.comped_objective(variables=call_args, samples=self.samples)
        print(result)
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


    def get_weights(self):
        return [self.w.value().numpy()]
