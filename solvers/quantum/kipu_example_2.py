from planqk.quantum.sdk import PlanqkQuantumProvider
from qiskit import QuantumCircuit, transpile

n_coin_tosses = 2

circuit = QuantumCircuit(n_coin_tosses)
for i in range(n_coin_tosses):
    circuit.h(i)
circuit.measure_all()

# Use the PLANQK CLI and log in with "planqk login" or set the environment variable PLANQK_PERSONAL_ACCESS_TOKEN.
# Alternatively, you can pass the access token as an argument to the constructor
provider = PlanqkQuantumProvider()
### If you do not want to login using the CLI, you can also provide your personal access token directly here:
# provider = PlanqkQuantumProvider(access_token="YOUR_PERSONAL_ACCESS_TOKEN_HERE")

# Select a quantum backend suitable for the task. All PLANQK supported quantum backends are
# listed at https://hub.kipu-quantum.com/quantum-backends.
backend = provider.get_backend("azure.ionq.simulator")

# Transpile the circuit ...
circuit = transpile(circuit, backend)
# ... and run it on the backend
job = backend.run(circuit, shots=100)

counts = job.result().get_counts()

print(counts)