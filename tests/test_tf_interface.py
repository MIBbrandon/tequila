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
@pytest.mark.parametrize("inputs", [np.random.uniform(0, np.pi*2, 3)])
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
    input_tensor = tf.convert_to_tensor(inputs)
    output = tensorflowed(input_tensor=input_tensor)
    summed = tf.math.reduce_sum(output)
    detached = tf.stop_gradient(summed).numpy()
    analytic = -np.sin(input_tensor[0]) + np.sin(input_tensor[1]) - np.sin(input_tensor[2])
    assert np.isclose(detached, analytic, atol=1.e-3)


@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("initial_values", [{'a': np.random.uniform(0, 2*np.pi), 'b': np.random.uniform(0, 2*np.pi)}])
def test_example_training(initial_values):
    U = tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs)
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    def train_step():
        # First, get a prediction
        pred = tensorflowed()
        # Then, calculate the loss of that prediction
        loss_value = tf.math.reduce_sum(pred).numpy()

        # TODO: how to mix loss_value with gradients

        # Get the gradients for just the parameters
        param_grads_values = tensorflowed.get_grads_values(only="params")

        # Adjust the parameters according to the gradient values
        optimizer.apply_gradients(zip(param_grads_values, [tensorflowed.get_angles()]))

    for i in range(STEPS):
        train_step()

    called = tf.math.reduce_sum(tensorflowed()).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)

@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("initial_values", [{'a': np.random.uniform(0, 2*np.pi), 'b': np.random.uniform(0, 2*np.pi)}])
@pytest.mark.parametrize("inputs", [np.random.uniform(0, np.pi*2, 2)])
def test_fixed_inputs(initial_values, inputs):
    U = tq.gates.Rx('c', 0) + tq.gates.Rx('d', 1) + tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
        + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['c', 'd'])
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    input_tensor = tf.Variable(inputs)

    def train_step():
        # First, get a prediction
        pred = tensorflowed(input_tensor=input_tensor)
        # Then, calculate the loss of that prediction
        loss_value = tf.math.reduce_sum(pred).numpy()

        # TODO: how to mix loss_value with gradients

        # Get the gradients for just the parameters
        param_grads_values = tensorflowed.get_grads_values(only="params")

        # Adjust the parameters according to the gradient values
        optimizer.apply_gradients(zip(param_grads_values, [tensorflowed.get_angles()]))

    for i in range(STEPS):
        train_step()

    called = tf.math.reduce_sum(tensorflowed(input_tensor=input_tensor)).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)

@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("initial_values", [{'a': np.random.uniform(0, 2*np.pi), 'b': np.random.uniform(0, 2*np.pi)}])
@pytest.mark.parametrize("inputs", [np.random.uniform(0, np.pi*2, 2)])
def test_fixed_params(initial_values, inputs):
    U = tq.gates.Rx('c', 0) + tq.gates.Rx('d', 1) + tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
        + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['c', 'd'])
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    input_tensor = tf.Variable(inputs)

    def train_step():
        # First, get a prediction
        pred = tensorflowed(input_tensor=input_tensor)
        # Then, calculate the loss of that prediction
        loss_value = tf.math.reduce_sum(pred).numpy()

        # TODO: how to mix loss_value with gradients

        # Get the gradients for just the inputs
        input_grads_values = tensorflowed.get_grads_values(only="inputs")

        # Adjust the inputs according to the gradient values
        optimizer.apply_gradients(zip(input_grads_values, [input_tensor]))

    for i in range(STEPS):
        train_step()

    called = tf.math.reduce_sum(tensorflowed(input_tensor=input_tensor)).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)

@pytest.mark.skipif(condition=not has_tf, reason="you don't have Tensorflow")
@pytest.mark.parametrize("initial_values", [{'a': np.random.uniform(0, 2*np.pi), 'b': np.random.uniform(0, 2*np.pi)}])
@pytest.mark.parametrize("inputs", [np.random.uniform(0, np.pi*2, 2)])
def test_no_fixed_var(initial_values, inputs):
    U = tq.gates.Rx('c', 0) + tq.gates.Rx('d', 1) + tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) \
        + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    tensorflowed = tq.ml.to_platform(stacked, platform='tensorflow', compile_args=cargs, input_vars=['c', 'd'])
    learning_rate = .1
    momentum = 0.9
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)

    input_tensor = tf.Variable(inputs)

    def train_step():
        # First, get a prediction
        pred = tensorflowed(input_tensor=input_tensor)
        # Then, calculate the loss of that prediction
        loss_value = tf.math.reduce_sum(pred).numpy()

        # TODO: how to mix loss_value with gradients

        # Get the gradients for both kinds of variables
        input_grads_values, param_grads_values = tensorflowed.get_grads_values()

        # Adjust the inputs and parameters according to the gradient values
        optimizer.apply_gradients(zip(input_grads_values, [input_tensor]))
        optimizer.apply_gradients(zip(param_grads_values, [tensorflowed.get_angles()]))

    for i in range(STEPS):
        train_step()

    called = tf.math.reduce_sum(tensorflowed(input_tensor=input_tensor)).numpy().tolist()
    assert np.isclose(called, 0.0, atol=1e-3)
