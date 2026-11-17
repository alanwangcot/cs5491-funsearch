# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080)
# Responsibility: LLM interface and strategy augmentation

"""Single-attempt DeepSeek HTTP client; the caller owns budget reservations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request


@dataclass
class GenerationResult:
    raw_text: str = ""
    status: str = "ok"
    error: str | None = None
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    request_id: str | None = None
    response_model: str | None = None
    finish_reason: str | None = None
    raw_response: dict | str | None = None
    http_status: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def read_api_key(env_path: str | Path | None = None) -> str:
    """Read only the named secret; never export it to worker environments."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = (
        Path(env_path)
        if env_path
        else Path(__file__).resolve().parents[3] / "project" / ".env"
    )
    if path.is_file():
        for line in path.read_text().splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() == "DEEPSEEK_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    raise ValueError(
        "DEEPSEEK_API_KEY is unavailable in the environment or configured .env file"
    )


class DeepSeekClient:
    """No automatic retries, redirects, model substitution, or key logging."""

    def __init__(
        self,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        env_path: str | Path | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        timeout: float = 90.0,
        thinking: str = "disabled",
        response_format: dict | None = None,
        endpoint: str | None = None,
        **extra,
    ):
        if extra:
            raise TypeError(f"unknown client options: {sorted(extra)}")
        self.model = model
        self.base_url = (endpoint or base_url).rstrip("/")
        self._api_key = api_key if api_key is not None else read_api_key(env_path)
        if not self._api_key:
            raise ValueError("API key cannot be empty")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.thinking = thinking
        self.response_format = response_format or {"type": "json_object"}

    @classmethod
    def from_config(cls, config: dict):
        options = dict(config.get("model", config))
        options.pop("provider", None)
        return cls(**options)

    def public_config(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "thinking": self.thinking,
            "response_format": self.response_format,
        }

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]")

    def generate(self, messages: list[dict]) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": self.thinking},
            "response_format": self.response_format,
            "stream": False,
        }
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        request = Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            # A custom opener disables redirects: one reservation means one physical request.
            import urllib.request

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None

            opener = urllib.request.build_opener(NoRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                body = self._redact(
                    response.read(2_000_000).decode("utf-8", errors="replace")
                )
                request_id = response.headers.get(
                    "x-request-id"
                ) or response.headers.get("request-id")
                http_status = response.status
            try:
                data = json.loads(body)
                choices = data["choices"]
                if len(choices) != 1:
                    raise ValueError("expected exactly one API choice")
                choice = choices[0]
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("response content is not text")
                return GenerationResult(
                    raw_text=content,
                    usage=data.get("usage", {}),
                    latency=time.monotonic() - started,
                    request_id=request_id or data.get("id"),
                    response_model=data.get("model"),
                    finish_reason=choice.get("finish_reason"),
                    raw_response=data,
                    http_status=http_status,
                )
            except (ValueError, KeyError, TypeError) as exc:
                return GenerationResult(
                    status="response_error",
                    error=str(exc),
                    raw_response=body,
                    request_id=request_id,
                    latency=time.monotonic() - started,
                    http_status=http_status,
                )
        except HTTPError as exc:
            body = self._redact(exc.read(100000).decode("utf-8", errors="replace"))
            return GenerationResult(
                status="fatal"
                if exc.code in {400, 401, 402, 403, 404}
                else "http_error",
                error=f"HTTP {exc.code}",
                raw_response=body,
                http_status=exc.code,
                request_id=exc.headers.get("x-request-id"),
                latency=time.monotonic() - started,
            )
        except (URLError, TimeoutError, OSError) as exc:
            return GenerationResult(
                status="transport_error",
                error=self._redact(str(exc)),
                latency=time.monotonic() - started,
            )


class FakeClient:
    """Deterministic offline test provider, with no network or credential dependency."""

    def __init__(self, responses: list[str | GenerationResult] | None = None):
        self.responses = responses
        self.calls = 0

    def public_config(self) -> dict:
        return {"provider": "fake", "model": "deterministic-offline-fixture"}

    def generate(self, messages: list[dict]) -> GenerationResult:
        index = self.calls
        self.calls += 1
        if self.responses is not None:
            response = self.responses[index % len(self.responses)]
            return (
                response
                if isinstance(response, GenerationResult)
                else GenerationResult(raw_text=response)
            )
        augmented = "Parent strategy:" in messages[-1]["content"]
        codes = [
            "def score_bins(item, remaining):\n    return -(remaining - item)\n",
            "def score_bins(item, remaining):\n    residual = remaining - item\n    return -residual - 0.1 * np.sqrt(residual)\n",
            "def score_bins(item, remaining):\n    return np.zeros_like(remaining)\n",
        ]
        response = {"code": codes[index % len(codes)]}
        if augmented:
            response["strategy"] = (
                "Favor the smallest residual capacity and prioritize exact fills."
            )
        return GenerationResult(
            raw_text=json.dumps(response),
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            response_model="deterministic-offline-fixture",
            finish_reason="stop",
        )
