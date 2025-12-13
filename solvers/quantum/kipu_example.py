# !pip install --upgrade planqk-service-sdk
from planqk.service.client import PlanqkServiceClient

consumer_key = "rl2y3oBuDidirbQG8LLS4vHx0sUa"
consumer_secret = "b0jkxiACjceEYH0zFS3tjiIwpksa"
service_endpoint = "https://gateway.platform.planqk.de/services/2992477d-23de-4f18-8450-50be44310a2b/execute"

# Create a client
client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)

# Prepare your service input
# Example request body extracted from the service's OpenAPI description
request = {
    "problem": {
        "(0, 7)": 0.5,
        "(0, 6)": 0.5,
        "(0, 5)": 0.5,
        "(1, 2)": 0.5,
        "(1, 3)": 0.5,
        "(1, 5)": 0.5,
        "(2, 3)": 0.5,
        "(2, 8)": 0.5,
        "(3, 8)": 0.5,
        "(4, 9)": 0.5,
        "(4, 6)": 0.5,
        "(4, 7)": 0.5,
        "(5, 8)": 0.5,
        "(6, 9)": 0.5,
        "(7, 9)": 0.5
    },
    "problem_type": "spin",
    "shots": 1000,
    "num_greedy_passes": 0,
    "return_circuit": False,
    "execute_circuit": True
}

# Start a service execution
service_execution = client.run(request=request)

# Wait for the service execution to finish (blocking)
service_execution.wait_for_final_state()

# Retrieve the result
result = service_execution.result()

print(result)