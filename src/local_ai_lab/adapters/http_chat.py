from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from local_ai_lab.adapters.base import AdapterUnavailableError, GenerationResult, InferenceAdapter, PathDescription, ProbeResult
from local_ai_lab.config import Workload
from local_ai_lab.evidence import Evidence, EvidenceKind, UNKNOWN
from local_ai_lab.metrics import TimingMetrics


class OpenAICompatibleAdapter(InferenceAdapter):
    base_url_env: str
    api_key_env: str
    default_url: str | None = None
    is_cloud = False

    def __init__(self, model=None) -> None:
        super().__init__(model)
        self.base_url = os.getenv(self.base_url_env, self.default_url or "").rstrip("/")
        self.api_key = os.getenv(self.api_key_env, "")

    def probe(self) -> ProbeResult:
        if not self.base_url:
            return ProbeResult(False, f"Set {self.base_url_env}.")
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=3,
            )
            return ProbeResult(
                response.is_success,
                f"HTTP {response.status_code} from /models",
                details=[f"Endpoint: {self.base_url}"],
                evidence=[Evidence("endpoint_reachable", str(response.is_success), EvidenceKind.OBSERVED, "HTTP /models")],
            )
        except httpx.HTTPError as exc:
            return ProbeResult(False, f"Endpoint unavailable: {exc}", details=[f"Endpoint: {self.base_url}"])

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def generate(self, workload: Workload) -> GenerationResult:
        if not self.base_url:
            raise AdapterUnavailableError(f"Set {self.base_url_env}.")
        model_id = self.model.id if self.model else os.getenv("LOCAL_AI_LAB_CLOUD_MODEL", "")
        if not model_id:
            raise AdapterUnavailableError("No model configured.")
        messages: list[dict[str, str]] = []
        if workload.system_prompt:
            messages.append({"role": "system", "content": workload.system_prompt})
        messages.append({"role": "user", "content": workload.prompt})
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": workload.generation.max_output_tokens,
            "temperature": workload.generation.temperature,
            "top_p": workload.generation.top_p,
        }
        if workload.generation.seed is not None:
            body["seed"] = workload.generation.seed

        start = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        server_metrics: dict[str, Any] = {}
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=300,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("timings"):
                    server_metrics = event["timings"]
                choices = event.get("choices") or []
                content = (choices[0].get("delta") or {}).get("content") if choices else None
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(content)
        finished = time.perf_counter()
        output_tokens = usage.get("completion_tokens")
        decode_seconds = finished - first_token_at if first_token_at else None
        return GenerationResult(
            text="".join(chunks),
            timing=TimingMetrics(
                ttft_seconds=first_token_at - start if first_token_at else None,
                decode_seconds=decode_seconds,
                streaming_seconds=decode_seconds,
                end_to_end_seconds=finished - start,
                prompt_tokens=usage.get("prompt_tokens"),
                output_tokens=output_tokens,
                decode_tokens_per_second=(output_tokens / decode_seconds) if output_tokens and decode_seconds else None,
                total_tokens_per_second=(usage.get("total_tokens") / (finished - start))
                if usage.get("total_tokens")
                else None,
            ),
            evidence=[
                Evidence("developer_api", "OpenAI-compatible HTTP", EvidenceKind.CONFIGURED, self.base_url),
                Evidence("execution_device", UNKNOWN, EvidenceKind.RUNTIME_REPORTED, "Server response did not identify a device"),
            ],
            usage=usage,
            server_metrics=server_metrics,
            warnings=["Client timing includes network transport."] if self.is_cloud else [],
        )
