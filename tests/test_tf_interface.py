import numpy as np
import pytest
import tequila as tq

has_tf = tq.HAS_TF
if has_tf:
    import tensorflow as tf
    from tensorflow.keras import optimizers

STEPS = 90


@pytest.mark.dependencies
def test_dependencies():
    assert has_tf


@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("inputs", [np.random.uniform(0, np.pi * 2, 3)])
def test_calls_correctly(inputs):
    U1 = tq.gates.Rx(angle='a', target=0)
    H1 = tq.paulis.Y(0)
    U2 = tq.gates.Ry(angle='b', target=0)
    H2 = tq.paulis.X(0)
    U3 = tq.gates.H(0) + tq.gates.Rz(angle='c', target=0) + tq.gates.H(0)
    H3 = tq.paulis.Y(0)

    evals = [tq.ExpectationValue(U1, H1), tq.ExpectationValue(U2, H2), tq.ExpectationValue(U3, H3)]
    stacked = tq.vectorize(evals)
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', input_vars=['a', 'b', 'c'])
    input_tensor = tf.constant(inputs, dtype=tf.float32)
    tensorflowed.set_input_values(input_tensor)
    output = tensorflowed()
    summed = tf.math.reduce_sum(output)
    detached = tf.stop_gradient(summed).numpy()
    analytic = -np.sin(input_tensor[0]) + np.sin(input_tensor[1]) - np.sin(input_tensor[2])
    assert np.isclose(detached, analytic, atol=1.e-3)


@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("initial_values",
                         [{'a': np.random.uniform(0, 2 * np.pi), 'b': np.random.uniform(0, 2 * np.pi)}])
def test_example_training(initial_values):
    U = tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    cargs = {'samples': None, 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs)
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    var_list_fn = lambda: tensorflowed.trainable_variables
    loss = lambda: tf.reduce_sum(tensorflowed())

    for i in range(STEPS):
        optimizer.minimize(loss, var_list_fn)

    called = tf.math.reduce_sum(tensorflowed()).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)


@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("initial_values",
                         [{'a': np.random.uniform(0, 2 * np.pi), 'b': np.random.uniform(0, 2 * np.pi)}])
@pytest.mark.parametrize("inputs", [np.random.uniform(0, np.pi * 2, 2)])
@pytest.mark.parametrize("fixed", ["inputs", "params", "nothing"])
def test_different_fixed_variables_cases(initial_values, inputs, fixed):
    U = tq.gates.Rx('c', 0) + tq.gates.Rx('d', 1) + tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
        + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['c', 'd'])
    initial_params = tensorflowed.get_params_list()
    input_tensor = tf.constant(inputs, dtype=tf.float32)
    tensorflowed.set_input_values(input_tensor)
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    if fixed == "inputs":
        var_list_fn = lambda: tensorflowed.get_params()  # Since we just want to train the parameters
    elif fixed == "params":
        var_list_fn = lambda: tensorflowed.get_inputs()  # Since we just want to train the inputs
    else:
        var_list_fn = lambda: tensorflowed.trainable_variables  # We train all trainable variables

    loss = lambda: tf.reduce_sum(tensorflowed())

    for i in range(STEPS):
        optimizer.minimize(loss, var_list_fn)

    called = tf.math.reduce_sum(tensorflowed()).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)

    # Check that what should have stayed fixed does stay fixed
    # NOTE: by chance, these assertions may fail because they happen to not need to change. Just run again to make sure
    if fixed == "inputs":
        assert all([np.isclose(before, after, atol=1e-3)
                    for before, after in zip(inputs, tensorflowed.get_inputs_list())])
    elif fixed == "params":
        assert all([np.isclose(before, after, atol=1e-3)
                    for before, after in zip(initial_params, tensorflowed.get_params_list())])
    else:
        assert not all([np.isclose(before, after, atol=1e-3)
                        for before, after in zip(inputs, tensorflowed.get_inputs_list())])
        assert not all([np.isclose(before, after, atol=1e-3)
                        for before, after in zip(initial_params, tensorflowed.get_params_list())])
