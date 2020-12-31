import numpy as np
import pytest
import tequila as tq

has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers

U = tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
H1 = tq.paulis.Qm(1)
H2 = tq.paulis.Qm(2)
H3 = tq.paulis.Qm(3)

tq.draw(U)

stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
stacked = tq.vectorize(stackable)

initial_values = {'a': 1.5, 'b': 2.}
cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs)
input_tensors = tensorflowed.get_weights()
learning_rate = .1
momentum = 0.9
expected_output_tensors = tf.constant([0, 0])
optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
train_acc = tf.keras.metrics.Mean()
train_loss = tf.keras.metrics.Mean()


# @tf.function
def train_step():
    # First, get a prediction
    pred = tensorflowed(0)
    # Then, calculate the loss of that prediction
    loss_value = tf.math.reduce_sum(pred).numpy()
    # Get the gradients
    grads = tensorflowed.get_weight_grads_values()

    print("\nAngles before: ", tensorflowed.angles.numpy().tolist())
    print("Loss: ", loss_value)
    print("Prediction: ", pred.numpy().tolist())
    print("Gradients: ", grads)
    # for j in range(len(grads)):
    #     for i in range(len(grads[j])):
    #         grads[j][i] *= -1
    print("Adjusted gradients: ", grads)
    print("w = " + str(tensorflowed.angles.numpy().tolist()) + " - " + str(learning_rate) + " * " + str(grads))

    optimizer.apply_gradients(zip(grads, [tensorflowed.angles]))

    print("Angles after: ", tensorflowed.angles.numpy().tolist(), "\n")


for i in range(200):
    train_step()

called = tf.math.reduce_sum(tensorflowed(0)).numpy().tolist()
assert np.isclose(called, 0.0, atol=1e-3)
