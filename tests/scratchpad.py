import numpy as np
import tequila as tq

has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers

U = tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
        + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
H1 = tq.paulis.Qm(1)
H2 = tq.paulis.Qm(2)
H3 = tq.paulis.Qm(3)

# tq.draw(U)

stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
stacked = tq.vectorize(stackable)

initial_values = {'a': 1.5, 'b': 2.}
cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs)
learning_rate = .1
momentum = 0.9
optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

# input_tensor = tf.Variable([1.5, 0.3])

# @tf.function
def train_step():
    # First, get a prediction
    pred = tensorflowed()
    # Then, calculate the loss of that prediction
    loss_value = tf.math.reduce_sum(pred).numpy()

    # TODO: how to mix loss_value with gradients

    # Get the gradients
    input_grads_values, param_grads_values = tensorflowed.get_grads_values()

    # print("\nInputs before: ", input_tensor.numpy().tolist())
    print("Angles before: ", tensorflowed.angles.numpy().tolist())
    print("Loss: ", loss_value)
    print("Prediction: ", pred.numpy().tolist())
    print("Gradients: ", param_grads_values)
    # for j in range(len(grads)):
    #     for i in range(len(grads[j])):
    #         grads[j][i] *= -1
    print("Adjusted gradients: ", param_grads_values)
    print("w = " + str(tensorflowed.angles.numpy().tolist()) + " - " + str(learning_rate) + " * " + str(param_grads_values))

    optimizer.apply_gradients(zip(param_grads_values, [tensorflowed.angles]))
    # optimizer.apply_gradients(zip(input_grads_values, [input_tensor]))
    # print("Inputs after: ", input_tensor.numpy().tolist())
    print("Angles after: ", tensorflowed.angles.numpy().tolist(), "\n")


for i in range(80):
    train_step()

called = tf.math.reduce_sum(tensorflowed()).numpy().tolist()
assert np.isclose(called, 0.0, atol=1e-3)
