from google.cloud import aiplatform


def create_endpoint_sample(
    project: str,
    display_name: str,
    location: str = "asia-south1",
    api_endpoint: str = "https://asia-south1-aiplatform.googleapis.com",
    timeout: int = 300,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)
    endpoint = {"display_name": display_name}
    parent = f"projects/{project}/locations/{location}"
    response = client.create_endpoint(parent=parent, endpoint=endpoint)
    print("Long running operation:", response.operation.name)
    create_endpoint_response = response.result(timeout=timeout)
    print("create_endpoint_response:", create_endpoint_response)
