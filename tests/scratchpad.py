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
cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['c', 'd'])
learning_rate = .1
momentum = 0.9
optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

input_tensor = tf.Variable([0., 0.])

# @tf.function
def train_step():
    with tf.GradientTape() as g:
        g.watch(input_tensor)
        # First, get a prediction
        pred = tensorflowed(input_tensor)

        # Then, calculate the loss of that prediction
        loss_value = tf.math.reduce_sum(pred)

    grads = g.gradient(loss_value, tensorflowed.trainable_variables)

    print("\nPrediction: ", pred.numpy().tolist())
    print("Loss: ", loss_value.numpy().tolist())
    print("Gradients: ", grads)
    print("Vars before: ", tensorflowed.trainable_variables)

    optimizer.apply_gradients(zip(grads, tensorflowed.trainable_variables))

    print("Vars after: ", tensorflowed.trainable_variables, "\n")
    # optimizer.minimize(loss_value, [tensorflowed.get_inputs(), tensorflowed.get_params()])



for i in range(80):
    train_step()

called = tf.math.reduce_sum(tensorflowed(input_tensor)).numpy().tolist()
assert np.isclose(called, 0.0, atol=1e-3)
