import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    _deployment_name: str
    _api_key: str

    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")
        self._deployment_name = deployment_name
        self._api_key = api_key

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int,
            print_request: bool = True,
            print_response: bool = False
    ) -> dict[int, list[float]]:
        if print_request:
            print(f"Searching similarities for `{inputs}` \nAnd such dimensions: {dimensions}\n📋Results:\n")

        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }

        request_data = {
            'input': inputs,
            'dimensions': dimensions,
        }

        response = requests.post(
            url=DIAL_EMBEDDINGS.format(model=self._deployment_name),
            headers=headers,
            json=request_data,
            timeout=60
        )

        if print_response:
            print("\n" + "=" * 50 + " RESPONSE" + "=" * 50)
            print(str(response.status_code) + ": " + response.text)

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        response_json = response.json()
        data = response_json.get('data', [])
        return self._from_data(data)

    def _from_data(self, data: list[dict]) -> dict[int, list[float]]:
        return {embedding_obj['index']: embedding_obj['embedding'] for embedding_obj in data}
