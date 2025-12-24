import networkx
from openqaoa.problems import MinimumVertexCover
from openqaoa import QAOA  

from openqaoa import QAOA, create_device
from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider

g = networkx.circulant_graph(6, [1])
vc = MinimumVertexCover(g, field=1.0, penalty=10)
qubo_problem = vc.qubo

# q = QAOA()
# q.compile(qubo_problem)
# q.optimize()

# print(q.result.optimized)
# q.result.plot_cost()

# from qiskit_ibm_runtime import QiskitRuntimeService
 
# QiskitRuntimeService.save_account(
# token="<your-api-key>", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
# instance="<CRN>", # Optional
# )

IBMProvider.save_account(token='Rb_LT59C4JFJ3HvGysonwDi9sWh1vQDRd-7YCbecD3nz', overwrite=True)

#Create the QAOA
q = QAOA()

qasm_sim_device = create_device(location='ibmq', 
                                name='ibmq_qasm_simulator',
                                hub='ibm-q', 
                                group='open', 
                                project='main')
q.set_device(qasm_sim_device)
q.compile(qubo_problem)

# Load account from disk
# provider = IBMProvider()
# backend  = provider.get_backend('ibmq_kolkata')

# Create a device
# ibmq_device = create_device(location='ibmq', name='ibmq_qasm_simulator')
# q.set_device(ibmq_device)

# q.set_circuit_properties(p=1, param_type='standard', init_type='ramp', mixer_hamiltonian='x')
# q.set_backend_properties(init_hadamard=True, n_shots=8000, cvar_alpha=0.85)
# q.set_classical_optimizer(method='cobyla', maxiter=50, tol=0.05)

q.optimize()