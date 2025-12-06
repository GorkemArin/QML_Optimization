# !pip install --upgrade planqk-service-sdk
from planqk.service.client import PlanqkServiceClient

consumer_key = "rl2y3oBuDidirbQG8LLS4vHx0sUa"
consumer_secret = "b0jkxiACjceEYH0zFS3tjiIwpksa"
service_endpoint = "https://gateway.platform.planqk.de/token"

# Create a client
client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)

# Prepare your service input
# Example request body extracted from the service's OpenAPI description
request = "{\n  \"problem\": {\n    \"(0, 7)\": 0.5,\n    \"(0, 6)\": 0.5,\n    \"(0, 5)\": 0.5,\n    \"(1, 2)\": 0.5,\n    \"(1, 3)\": 0.5,\n    \"(1, 5)\": 0.5,\n    \"(2, 3)\": 0.5,\n    \"(2, 8)\": 0.5,\n    \"(3, 8)\": 0.5,\n    \"(4, 9)\": 0.5,\n    \"(4, 6)\": 0.5,\n    \"(4, 7)\": 0.5,\n    \"(5, 8)\": 0.5,\n    \"(6, 9)\": 0.5,\n    \"(7, 9)\": 0.5,\n    \"()\": -7.5\n  },\n  \"problem_type\": \"spin\",\n  \"shots\": 1000,\n  \"num_greedy_passes\": 0,\n  \"return_circuit\": False,\n  \"execute_circuit\": True\n}"

# Start a service execution
service_execution = client.run(request=request)

# Wait for the service execution to finish (blocking)
service_execution.wait_for_final_state()

# Retrieve the result
result = service_execution.result()

print(result)