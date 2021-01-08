from collections import OrderedDict

import numpy as np
import tequila as tq

has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.losses import mse

U = tq.gates.Rx('c', 0) + tq.gates.Rx('d', 1) + tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
    + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
H1 = tq.paulis.Qm(1)
H2 = tq.paulis.Qm(2)
H3 = tq.paulis.Qm(3)

# tq.draw(U)

stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
stacked = tq.vectorize(stackable)
initial_values = {'a': 1.5, 'b': 2.}
initial_input_values = {'d': 2., 'c': 1.8}
cargs = {'samples': None, 'initial_values': initial_values}
tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['d', 'c'])
tensorflowed.set_input_values(initial_input_values)
learning_rate = .1
momentum = 0.9
optimizer = optimizers.Adam(lr=learning_rate)

desired_output = tf.constant([0., 0., 0.])

var_list_fn = lambda: tensorflowed.trainable_variables

loss = lambda: mse(tensorflowed(), desired_output)

print("Before training: ", tensorflowed.get_input_values(), tensorflowed.get_params_values())

for i in range(100):
    print(tensorflowed.get_input_values(), tensorflowed.get_params_values())
    optimizer.minimize(loss, var_list_fn)

called = tf.math.reduce_sum(tensorflowed()).numpy().tolist()
print("Final prediction: ", tensorflowed().numpy().tolist())
print("Final loss: ", called)
print("Final variable values: ", tensorflowed.get_input_values(), tensorflowed.get_params_values())
