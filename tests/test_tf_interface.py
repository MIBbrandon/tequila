import numpy as np
import pytest
import tequila as tq

has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers

@pytest.mark.dependencies
def test_dependencies():
    assert has_tf

# TODO: currently under review to make this test work
# @pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
# @pytest.mark.parametrize("angles", [np.random.uniform(0, np.pi*2, 3)])
# def test_calls_correctly(angles: tf.Tensor):
#     U1 = tq.gates.Rx(angle='a', target=0)
#     H1 = tq.paulis.Y(0)
#     U2 = tq.gates.Ry(angle='b', target=0)
#     H2 = tq.paulis.X(0)
#     U3 = tq.gates.H(0) + tq.gates.Rz(angle='c', target=0) + tq.gates.H(0)
#     H3 = tq.paulis.Y(0)
#
#     evals = [tq.ExpectationValue(U1, H1), tq.ExpectationValue(U2, H2), tq.ExpectationValue(U3, H3)]
#     stacked = tq.vectorize(evals)
#     tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', input_vars=['a', 'b', 'c'])
#     inputs = tf.convert_to_tensor(angles)
#     output = tensorflowed(inputs)
#     summed = tf.math.reduce_sum(output)
#     detached = tf.stop_gradient(summed).numpy()
#     analytic = -np.sin(angles[0]) + np.sin(angles[1]) - np.sin(angles[2])
#     assert np.isclose(detached, analytic, atol=1.e-3)


@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
def test_example_training():
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
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    # @tf.function
    def train_step():
        # First, get a prediction
        pred = tensorflowed(0)  # 0 is ignored, Layer requires input no matter what, still trying to find a workaround
        # Then, calculate the loss of that prediction
        loss_value = tf.math.reduce_sum(pred).numpy()

        # TODO: how to mix loss_value with gradients

        # Get the gradients
        grads = tensorflowed.get_grads_values()

        optimizer.apply_gradients(zip(grads, [tensorflowed.angles]))

    for i in range(80):
        train_step()

    called = tf.math.reduce_sum(tensorflowed(0)).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)
