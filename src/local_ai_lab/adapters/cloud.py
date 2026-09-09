from __future__ import annotations

from local_ai_lab.adapters.base import PathDescription, ProbeResult
from local_ai_lab.adapters.http_chat import OpenAICompatibleAdapter
from local_ai_lab.evidence import UNKNOWN


class CloudAdapter(OpenAICompatibleAdapter):
    id = "cloud"
    base_url_env = "LOCAL_AI_LAB_CLOUD_BASE_URL"
    api_key_env = "LOCAL_AI_LAB_CLOUD_API_KEY"
    default_url = "https://api.openai.com/v1"
    is_cloud = True

    def probe(self) -> ProbeResult:
        if not self.api_key:
            return ProbeResult(False, "Set LOCAL_AI_LAB_CLOUD_API_KEY; the endpoint was not contacted.")
        return super().probe()

    def describe(self) -> PathDescription:
        return PathDescription(
            id=self.id,
            name="OpenAI-compatible cloud inference",
            abstraction_level="Remote developer API",
            stack=["Application", "OpenAI-compatible HTTP API", UNKNOWN, UNKNOWN, UNKNOWN, "Provider-managed model artifact"],
            responsibilities={
                "model owner": "cloud provider",
                "model distribution": "cloud provider",
                "runtime": UNKNOWN,
                "inference engine": UNKNOWN,
                "backend": UNKNOWN,
                "hardware": UNKNOWN,
                "hardware selection": "cloud provider",
                "tokenization": "cloud service",
                "KV cache": "cloud service",
                "sampling": "cloud service, configured by application parameters",
                "prefill/decode": "cloud service",
                "offline": "no",
                "application ships": "client code and configuration; never credentials",
                "Windows responsibility": "networking and client process",
                "application responsibility": "request, credentials, validation, client-observed timing",
            },
            you_control=["prompt", "published model identifier", "generation parameters"],
            runtime_controls=["runtime, backend, hardware, batching, caching"],
            windows_controls=["client networking and process scheduling"],
            model_format=UNKNOWN,
            notes=["Client-observed latency includes transport and queueing. It is not pure model compute."],
        )
