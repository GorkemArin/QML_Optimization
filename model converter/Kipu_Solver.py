from planqk.service.client import PlanqkServiceClient

consumer_key = "rl2y3oBuDidirbQG8LLS4vHx0sUa"
consumer_secret = "b0jkxiACjceEYH0zFS3tjiIwpksa"
# service_endpoint = "https://gateway.platform.planqk.de/kipu-quantum/miray-advanced-quantum-optimizer---simulator/1.0.0"
service_endpoint_illay = "https://gateway.platform.planqk.de/kipu-quantum/illay-base-quantum-optimizer/1.0.0"
service_endpoint_miray = "https://gateway.platform.planqk.de/kipu-quantum/miray-advanced-quantum-optimizer---simulator/1.0.0"

str_to_int = {}
int_to_str = {}

def encode_string(s):
    if s not in str_to_int:
        idx = len(str_to_int)
        str_to_int[s] = idx
        int_to_str[idx] = s
    return str_to_int[s]

def get_problem_qubo(qubo:dict) -> dict:
    new_d = {}
    for (a, b), v in qubo.items():
        i = encode_string(a)
        j = encode_string(b)
        if i > 19 or j > 19:
            continue
        elif i == j:
            new_d[f"({i},)"] = v
        else:
            new_d[f"({i}, {j})"] = v
    return new_d

def get_result_dict(result):
    global int_to_str
    decoded = {}
    for k, v in result.items():
        # i, j = map(int, k.strip("()").split(", "))
        decoded[int_to_str[int(k)]] = v
    return decoded

def solve_kipu(qubo: dict, solver: str) -> dict:
# Create a client
    problem = get_problem_qubo(qubo)
    print('problem:', problem)
    print('type:', type(problem))
    if solver == 'kipu-illay':
        client = PlanqkServiceClient(service_endpoint_illay, consumer_key, consumer_secret)
        request = \
        {
            "problem": problem,
            "problem_type": "binary",
            "shots": 100,
            "num_greedy_passes": 0,
            "return_circuit": False,
            "execute_circuit": True
        }
    elif solver == 'kipu-miray':
        client = PlanqkServiceClient(service_endpoint_miray, consumer_key, consumer_secret)
        request = \
        {
            "problem": problem,
            "problem_type": "binary",
            "shots": 100,
            "num_iterations": 3,
            "num_greedy_passes": 0
        }

    # Start a service execution
    service_execution = client.run(request=request)

    print('Waiting for Kipu service execution...')
    # Wait for the service execution to finish (blocking)
    service_execution.wait_for_final_state()

    # Retrieve the result
    result = service_execution.result()
    if(result.embedded.status.status == 'FAILED'):
        print('Result:', result)
        raise Exception('Kipu Solver failed!')
    
    extra = result.model_extra
    mapped_solution = extra['result']['mapped_solution']

    return get_result_dict(mapped_solution)