from qiskit_optimization import QuadraticProgram

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
qp = QuadraticProgram()

# number of variables
n = 10
for i in range(n):
    qp.binary_var(name=f"x{i}")

# edges (i, j): weight
edges = {
    (0, 7): 0.5,
    (0, 6): 0.5,
    (0, 5): 0.5,
    (1, 2): 0.5,
    (1, 3): 0.5,
    (1, 5): 0.5,
    (2, 3): 0.5,
    (2, 8): 0.5,
    (3, 8): 0.5,
    (4, 9): 0.5,
    (4, 6): 0.5,
    (4, 7): 0.5,
    (5, 8): 0.5,
    (6, 9): 0.5,
    (7, 9): 0.5,
}

# Max-Cut objective
for (i, j), w in edges.items():
    qp.minimize(quadratic={(f"x{i}", f"x{j}"): w})

print(qp)

ising, offset = qp.to_ising()

qaoa = QAOA(
    sampler=Sampler(),
    optimizer=COBYLA(),
    reps=2
)

result = qaoa.compute_minimum_eigenvalue(ising)

x = ising[0].num_qubits
bitstring = result.eigenstate.binary_probabilities().most_likely()

print("Best bitstring:", bitstring)
print("Energy:", result.eigenvalue + offset)
