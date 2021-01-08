import numpy as np
import tequila as tq

has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers

U = tq.gates.Rx('c', 0) + tq.gates.Rx('d', 1) + tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
    + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
H1 = tq.paulis.Qm(1)
H2 = tq.paulis.Qm(2)
H3 = tq.paulis.Qm(3)

# tq.draw(U)

stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
stacked = tq.vectorize(stackable)

initial_values = {'a': 1.5, 'b': 2.}
# input_tensor = tf.constant([2., 1.8], dtype=tf.float32)
cargs = {'samples': None}
tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['a', 'b'])
# tensorflowed.set_input_values(input_tensor)
learning_rate = .1
momentum = 0.9
optimizer = optimizers.Adam(lr=learning_rate)

var_list_fn = lambda: tensorflowed.trainable_variables

loss = lambda: tf.reduce_sum(tensorflowed())

for i in range(200):
    print([x.numpy().tolist() for x in tensorflowed.trainable_variables])
    optimizer.minimize(loss, var_list_fn)

called = tf.math.reduce_sum(tensorflowed()).numpy().tolist()
print("Final prediction: ", tensorflowed().numpy().tolist())
print("Final loss: ", called)
print("Final variable values: ", [x.numpy().tolist() for x in tensorflowed.trainable_variables])
