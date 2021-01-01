import tequila as tq
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.optimizers as optims

bond_lengths = np.linspace(.3,1.6,20) # our bond length, in angstrom.
amp_arrays = []
state_preps = []
for i in bond_lengths:
    # the line below initializes a tequila molecule object for H2 at a specific bond length.
    # see the quantum chemistry tutorial for more details.
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\n H 0.0 0.0 {}".format(str(i)), basis_set="sto-3g")
    amplitude = molecule.compute_amplitudes(method='ccsd') # get the state prep amplitudes
    amp_arrays.append(np.asarray([v for v in amplitude.make_parameter_dictionary().values()]))
    state_preps.append(molecule.make_uccsd_ansatz(trotter_steps=1,initial_amplitudes=amplitude))

def data_generator(amp_arrays):
    i = 0
    while i < len(amp_arrays):
        yield amp_arrays[i]
        i += 1

# for n in data_generator(amp_arrays):
#     print(n)
#     print(type(n))

# TODO: consider determining the output_shapes
my_data = tf.data.Dataset.from_generator(data_generator, args=[amp_arrays], output_types=tf.float64)



encoder = tq.gates.Rx('a',0) +tq.gates.Rx('b',1) +tq.gates.CNOT(1,3) +tq.gates.CNOT(0,2)+tq.gates.CNOT(0,1)
state_prep = state_preps[0] # every member of this list is the same object; it doesn't matter which we pick.
combined = state_prep + encoder
print('combined state prep, encoder circuit:  \n', combined)

# we decide that the 3rd and 4th qubits will be trash qubits. The hamiltonian below projects onto zero.
hamiltonian = tq.hamiltonian.paulis.Qm(2)*tq.hamiltonian.paulis.Qm(3)
h2_encoder = tq.ExpectationValue(U=combined,H=hamiltonian)
print('H2 autoencoder: ', h2_encoder)

input_variable=h2_encoder.extract_variables()[0]
# inits={'a':1.5, 'b':0.5}
inits={'a':np.random.uniform(0, 2*np.pi), 'b':np.random.uniform(0, 2*np.pi)}
compile_args={'backend':'qulacs', 'initial_values':inits, 'samples': 10000} # dict. allowed keys: backend, samples, noise, device, initial_values

my_tf_encoder = tq.ml.to_platform(h2_encoder, platform='tensorflow', compile_args=compile_args, input_vars=[input_variable])
# print(my_tf_encoder)

optim = optims.SGD(lr=0.01, momentum=0.9)
loss_values = []

num_epochs = 30

for epoch in range(num_epochs):
    print('*** Epoch {} ***'.format(epoch+1))
    batch = my_data.shuffle(20).take(10)
    batch_loss = []
    for point in batch:
        pred = my_tf_encoder(point)
        loss = tf.math.reduce_mean(pred)
        batch_loss.append(loss)

        # Get the gradients and apply them
        param_grads_values = my_tf_encoder.get_grads_values(only="params")
        optim.apply_gradients(zip(param_grads_values, [my_tf_encoder.get_angles()]))
        print("\t\tPrediction: ", pred.numpy())
        print("\t\tLoss: ", loss.numpy())
    bv = np.mean([l.numpy() for l in batch_loss])
    loss_values.append(bv)
    print('\tBatched Average Loss: ', bv, "\n")

print("Final parameter values: ")
print(my_tf_encoder.get_angles().numpy())

plt.plot(loss_values, label='loss per epoch')
plt.legend()
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Autoencoder Loss', fontsize=16)
plt.show()
