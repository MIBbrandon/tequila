import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import matplotlib.pyplot as plt

import tequila as tq
from tequila.ml.utils_ml import TequilaMLException, preamble
from tequila.objective import Objective, vectorize
from tequila.tools import list_assignment

# Circuit
# U = tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
U = tq.gates.Y(power='a', target=0)

# Hamiltonians
H1 = tq.paulis.X(0)
# H1 = tq.paulis.Qm(1)
# H2 = tq.paulis.Qm(2)
# H3 = tq.paulis.Qm(3)

# stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
stackable = [tq.ExpectationValue(U, H1)]
stacked = tq.vectorize(stackable)

# compiled_objective = tq.compile(E, backend='cirq')
objective = stacked  # VectorObjective
# initial_values = {'a': 1.5, 'b': 2.}
initial_values = {'a': 1.5}
compile_args = {'samples': None, 'backend': 'cirq', 'initial_values': initial_values}
input_vars = None
if isinstance(objective, tuple) or isinstance(objective, list) or isinstance(objective, Objective):
    objective = vectorize(list_assignment(objective))
compiled_objective, compile_args, weight_vars, w_grads, i_grads, first, second \
    = preamble(objective, compile_args, input_vars)
samples = compile_args['samples']

def getCircuit(compiled_objective) -> cirq.Circuit:
    return compiled_objective.args[0].U.circuit

def getQubits(compiled_objective):
    q_map = compiled_objective.args[0].U.qubit_map
    qs = []
    for qubit_num in q_map:
        qs.append(q_map[qubit_num].instance)
    return qs

def getCircuits(compiled_objective):
    circuits = []
    operators = []
    sympy_to_tq = []
    tq_to_sympy = []
    for compiled_individual_objective in compiled_objective.args:
        # We first get the circuits of our objectives
        circuits.append(compiled_individual_objective.U.circuit)
        # Now we also want the operator
        operators.append(compiled_individual_objective.H[0].qubit_operator)
        # We also want the dictionaries to have the names of the parameters
        sympy_to_tq.append(compiled_individual_objective.U.sympy_to_tq)
        tq_to_sympy.append(compiled_individual_objective.U.tq_to_sympy)
    return circuits, operators, sympy_to_tq, tq_to_sympy

# TODO: get the Cirq operators used. Neither QubitHamiltonian nor QubitOperator are valid, since they are too abstract

# For now, we will just use our own operators until the previous todoThing is solved
circuits, operators, sympy_to_tq, tq_to_sympy = getCircuits(compiled_objective)
qubits = getQubits(compiled_objective)
temp_operator = cirq.X(qubits[0])
temp_operators = [cirq.X(qubits[0])]

names = [tq_to_sympy[0][name] for name in initial_values]
print(names)
# values = [[initial_values[name]] for name in initial_values]
values = np.linspace(0, 5, 200)[:, np.newaxis].astype(np.float32)
# print(values)



# Differentiators
expectation_calculation = tfq.layers.Expectation(
    differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01))

sampled_expectation_calculation = tfq.layers.SampledExpectation(
    differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01))

gradient_safe_sampled_expectation = tfq.layers.SampledExpectation(
    differentiator=tfq.differentiators.ParameterShift())


# Make input_points = [batch_size, 1] array.
input_points = np.linspace(0, 5, 200)[:, np.newaxis].astype(np.float32)
exact_outputs = expectation_calculation(circuits[0],
                                        operators=temp_operator,
                                        symbol_names=names,
                                        symbol_values=input_points)
imperfect_outputs = sampled_expectation_calculation(circuits[0],
                                                    operators=temp_operator,
                                                    repetitions=500,
                                                    symbol_names=names,
                                                    symbol_values=input_points)
plt.title('Forward Pass Values')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.plot(input_points, exact_outputs, label='Analytic')
plt.plot(input_points, imperfect_outputs, label='Sampled')
plt.legend()
plt.show()

# Gradients are a much different story.
values_tensor = tf.convert_to_tensor(input_points)


with tf.GradientTape() as g:
    g.watch(values_tensor)
    exact_outputs = expectation_calculation(circuits[0],
                                                    operators=temp_operator,
                                                    symbol_names=names,
                                                    symbol_values=values_tensor)
analytic_finite_diff_gradients = g.gradient(exact_outputs, values_tensor)

with tf.GradientTape() as g:
    g.watch(values_tensor)
    imperfect_outputs = sampled_expectation_calculation(
        circuits[0],
        operators=temp_operator,
        repetitions=500,
        symbol_names=names,
        symbol_values=values_tensor)
sampled_finite_diff_gradients = g.gradient(imperfect_outputs, values_tensor)

plt.title('Gradient Values')
plt.xlabel('$x$')
plt.ylabel('$f^{\'}(x)$')
plt.plot(input_points, analytic_finite_diff_gradients, label='Analytic')
plt.plot(input_points, sampled_finite_diff_gradients, label='Sampled')
plt.legend()
plt.show()

with tf.GradientTape() as g:
    g.watch(values_tensor)
    imperfect_outputs = gradient_safe_sampled_expectation(
        circuits[0],
        operators=temp_operator,
        repetitions=500,
        symbol_names=names,
        symbol_values=values_tensor)

sampled_param_shift_gradients = g.gradient(imperfect_outputs, values_tensor)

plt.title('Gradient Values')
plt.xlabel('$x$')
plt.ylabel('$f^{\'}(x)$')
plt.plot(input_points, analytic_finite_diff_gradients, label='Analytic')
plt.plot(input_points, sampled_param_shift_gradients, label='Sampled')
plt.legend()
plt.show()





