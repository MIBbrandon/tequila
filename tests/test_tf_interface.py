import numpy as np
import pytest
import tequila as tq
has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers

np.random.seed(15)

@pytest.mark.dependencies
def test_dependencies():
    assert has_tf


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

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    initial_values = {'a': 1.5, 'b': 2.}
    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs)
    input_tensors = tensorflowed.get_weights()
    expected_output_tensors = tf.constant([0, 0])
    optimizer = optimizers.SGD(tensorflowed.get_weights(), lr=.1, momentum=0.9)
    loss_fn = tf.keras.metrics.Sum()
    train_acc = tf.keras.metrics.Mean()
    train_loss = tf.keras.metrics.Mean()

    # @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = tensorflowed(x)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, tensorflowed.trainable_weights)
        optimizer.apply_gradients(zip(grads, tensorflowed.trainable_weights))
        acc_value = tf.math.equal(y, tf.math.round(tf.keras.activations.sigmoid(logits)))
        train_acc.update_state(acc_value)
        train_loss.update_state(loss_value)

    for i in range(80):
        train_step(input_tensors, expected_output_tensors)

    called = tensorflowed().sum().detach().numpy()
    assert np.isclose(called, 0.0, atol=1e-3)