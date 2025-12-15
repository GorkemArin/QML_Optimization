from planqk.service.client import PlanqkServiceClient

consumer_key = "rl2y3oBuDidirbQG8LLS4vHx0sUa"
consumer_secret = "b0jkxiACjceEYH0zFS3tjiIwpksa"
# service_endpoint = "https://gateway.platform.planqk.de/kipu-quantum/miray-advanced-quantum-optimizer---simulator/1.0.0"
service_endpoint = "https://gateway.platform.planqk.de/kipu-quantum/illay-base-quantum-optimizer/1.0.0"

# Create a client
client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)

# Prepare your service input
# Example request body extracted from the service's OpenAPI description
request = {
    "problem": {
            "(zero, seven)": 0.5,
            "(zero, six)": 0.5,
            "(zero, five)": 0.5,
            "(one, two)": 0.5,
            "(one, three)": 0.5,
            "(one, five)": 0.5,
            "(two, three)": 0.5,
            "(two, eight)": 0.5,
            "(three, eight)": 0.5,
            "(four, nine)": 0.5,
            "(four, six)": 0.5,
            "(four, seven)": 0.5,
            "(five, eight)": 0.5,
            "(six, nine)": 0.5,
            "(seven, nine)": 0.5
        },
        "problem_type": "spin",
        "shots": 1000,
        "num_greedy_passes": 0,
        "return_circuit": True,
        "execute_circuit": True
    }

problem = {
            "(0, 7)": 0.54541234,
            "(0, 6)": -0.54542545,
            "(0, 9)": -0.5657641,
            "(6,)": -1.5654376234,
            "(5, 12)": 0.54541234,
            "(5, 18)": -0.54542545,
            "(5, 14)": -0.5657641,
            "(11,)": -1.5654376234,
            "(7, 4)": 0.54541234,
            "(7, 16)": -0.54542545,
            "(7, 10)": -0.5657641,
            "(3,)": -1.5654376234,
            "(1, 5)": 0.54541234,
            "(1, 8)": -0.54542545,
            "(1, 10)": -0.5657641,
            "(6,)": -1.5654376234,
            "(2, 13)": 0.54541234,
            "(2, 8)": -0.54542545,
            "(2, 9)": -0.5657641,
            "(5,)": -1.5654376234,
            "(7, 10)": 0.54541234,
            "(7, 11)": -0.54542545,
            "(7, 12)": -0.5657641,
            "(5,)": -1.5654376234
        }

print(problem)
print('type:', type(problem))

request_2 = {
    "problem": problem,
        "problem_type": "binary",
        "shots": 1000,
        "num_greedy_passes": 0,
        "return_circuit": True,
        "execute_circuit": True
    }

# Start a service execution
service_execution = client.run(request=request_2)

# Wait for the service execution to finish (blocking)
service_execution.wait_for_final_state()

# Retrieve the result
result = service_execution.result()
extra = result.model_extra
if(result.embedded.status.status == 'FAILED'):
    print('Result:', result)
    raise Exception('Kipu Solver failed!')

mapped_solution = extra['result']['mapped_solution']

print('Result:', mapped_solution)

'''
EXAMPLE
Result: links=ResultResponseLinks(status=HalLink(href='/7b2dc423-1193-4521-8d5a-efaaca2756de', templated=None, type=None,
 deprecation=None, name=None, profile=None, title=None, hreflang=None), self={'href': '/7b2dc423-1193-4521-8d5a-efaaca2756de/results'}) 
 embedded=ResultResponseEmbedded(status=ServiceExecution(id='7b2dc423-1193-4521-8d5a-efaaca2756de', created_at='2025-12-15 16:36:31',
   started_at='2025-12-15 16:36:32', ended_at='2025-12-15 16:38:26', status='SUCCEEDED', type='WORKFLOW',
     service_id='652b093d-8e84-46e6-9054-815665a43df7', service_definition_id='4e7919d4-7509-455c-ba28-3614fad695ff', 
     application_id='ee7a1147-15bb-4fa0-a2ae-94b557fbb7ea')) result={'cost': -13, 'bitstring': '0110110010', 'mapped_solution':
       {'8': 1, '4': 1, '9': 1, '5': -1, '0': 1, '2': -1, '7': -1, '3': -1, '6': -1, '1': 1}} 
       mapping={'8': 6, '4': 2, '9': 8, '5': 3, '0': 0, '2': 7, '7': 4, '3': 9, '6': 1, '1': 5} 
       circuit_info={'gate_count': 230, 'operations': {'s': 30, 'cx': 60, 'rz': 30, 'h': 70, 'sdg': 30, 'measure': 10},
         'depth': 46, 'num_qubits': 10, 'width': 20, 'circuit_string': 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[10];\ncreg c[10];
         \nh q[0];\nh q[1];\nh q[2];\nh q[3];\nh q[4];\nh q[5];\nh q[6];\nh q[7];\nh q[8];\nh q[9];\nsdg q[6];\nh q[6];\ncx q[1],q[6];
         \nrz(0.2966717789511306) q[6];\ncx q[1],q[6];\nh q[6];\ns q[6];\nsdg q[1];\nh q[1];\ncx q[1],q[6];\nrz(0.2966717789511306) q[6];
         \ncx q[1],q[6];\nh q[1];\ns q[1];\nsdg q[8];\nh q[8];\ncx q[7],q[8];\nrz(0.2966717789511306) q[8];\ncx q[7],q[8];\nh q[8];\ns q[8];
         \nsdg q[7];\nh q[7];\ncx q[7],q[8];\nrz(0.2966717789511306) q[8];\ncx q[7],q[8];\nh q[7];\ns q[7];\nsdg q[9];\nh q[9];\ncx q[5],q[9];
         \nrz(0.2966717789511306) q[9];\ncx q[5],q[9];\nh q[9];\ns q[9];\nsdg q[5];\nh q[5];\ncx q[5],q[9];\nrz(0.2966717789511306) q[9];
         \ncx q[5],q[9];\nh q[5];\ns q[5];\nsdg q[2];\nh q[2];\ncx q[0],q[2];\nrz(0.2966717789511306) q[2];\ncx q[0],q[2];\nh q[2];\ns [8];
         \ncx q[3],q[8];\nh q[8];\ns q[8];\nsdg q[3];\nh q[3];\ncx q[3],q[8];\nrz(0.2966717789511306) q[8];\ncx q[3],q[8];\nh q[3];\ns q[3];
         \nsdg q[3];\nh q[3];\ncx q[2],q[3];\nrz(0.2966717789511306) q[3];\ncx q[2],q[3];\nh q[3];\ns q[3];\nsdg q[2];\nh q[2];\ncx q[2],q[3];
         \nrz(0.2966717789511306) q[3];\ncx q[2],q[3];\nh q[2];\ns q[2];\nsdg q[5];\nh q[5];\ncx q[4],q[5];\nrz(0.2966717789511306) q[5];
         \ncx q[4],q[5];\nh q[5];\ns q[5];\nsdg q[4];\nh q[4];\ncx q[4],q[5];\nrz(0.2966717789511306) q[5];\ncx q[4],q[5];\nh q[4];\ns q[4];
         \nsdg q[9];\nh q[9];\ncx q[1],q[9];\nrz(0.2966717789511306) q[9];\ncx q[1],q[9];\nh q[9];\ns q[9];\nsdg q[1];\nh q[1];\ncx q[1],q[9];
         \nrz(0.2966717789511306) q[9];\ncx q[1],q[9];\nh q[1];\ns q[1];\nsdg q[8];\nh q[8];\ncx q[0],q[8];\nrz(0.2966717789511306) q[8];
         \ncx q[0],q[8];\nh q[8];\ns q[8];\nsdg q[0];\nh q[0];\ncx q[0],q[8];\nrz(0.2966717789511306) q[8];\ncx q[0],q[8];\nh q[0];\ns q[0];
         \nsdg q[7];\nh q[7];\ncx q[3],q[7];\nrz(0.2966717789511306) q[7];\ncx q[3],q[7];\nh q[7];\ns q[7];\nsdg q[3];\nh q[3];\ncx q[3],q[7];
         \nrz(0.2966717789511306) q[7];\ncx q[3],q[7];\nh q[3];\ns q[3];\nsdg q[6];\nh q[6];\ncx q[4],q[6];\nrz(0.2966717789511306) q[6];
         \ncx q[4],q[6];\nh q[6];\ns q[6];\nsdg q[4];\nh q[4];\ncx q[4],q[6];\nrz(0.2966717789511306) q[6];\ncx q[4],q[6];\nh q[4];\ns q[4];
         \nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\nmeasure q[4] -> c[4];\nmeasure q[5] -> c[5];
         \nmeasure q[6] -> c[6];\nmeasure q[7] -> c[7];\nmeasure q[8] -> c[8];\nmeasure q[9] -> c[9];'}
'''