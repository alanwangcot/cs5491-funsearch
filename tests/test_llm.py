# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080)
# Responsibility: LLM interface and strategy augmentation

import io
import json
from urllib.error import HTTPError, URLError

from funsearch_project.llm import DeepSeekClient, FakeClient, read_api_key


class Response:
    headers = {"x-request-id": "req-fixture"}
    status = 200

    def __init__(self, data):
        self.data = json.dumps(data).encode()

    def read(self, limit):
        return self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_single_attempt_with_matching_settings_and_metadata(monkeypatch):
    requests = []
    data = {
        "id": "completion-id",
        "model": "deepseek-flash",
        "choices": [
            {"message": {"content": '{"code":"fixture"}'}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }

    class Opener:
        def open(self, req, timeout):
            requests.append((req, timeout))
            return Response(data)

    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Opener())
    client = DeepSeekClient(api_key="fake-sensitive-secret")
    result = client.generate([{"role": "user", "content": "JSON"}])
    assert len(requests) == 1
    payload = json.loads(requests[0][0].data)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert result.status == "ok"
    assert result.request_id == "req-fixture"
    assert result.usage["completion_tokens"] == 8
    assert result.response_model == "deepseek-flash"
    assert "fake-sensitive-secret" not in str(client.public_config())


def test_error_does_not_retry_and_redacts(monkeypatch):
    calls = []

    class Opener:
        def open(self, req, timeout):
            calls.append(req)
            raise HTTPError(
                req.full_url, 401, "unauthorized", {}, io.BytesIO(b"fake-secret")
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Opener())
    result = DeepSeekClient(api_key="fake-secret").generate([])
    assert len(calls) == 1
    assert result.status == "fatal"
    assert result.raw_response == "[REDACTED]"


def test_transport_error_is_one_attempt(monkeypatch):
    calls = []

    class Opener:
        def open(self, req, timeout):
            calls.append(req)
            raise URLError("fixture network failure")

    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Opener())
    assert DeepSeekClient(api_key="fixture").generate([]).status == "transport_error"
    assert len(calls) == 1


def test_env_key_read_without_export(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text('DEEPSEEK_API_KEY="fixture-key"\n')
    assert read_api_key(p) == "fixture-key"
    import os

    assert "DEEPSEEK_API_KEY" not in os.environ


def test_fake_provider_no_credentials():
    fake = FakeClient()
    assert "strategy" not in json.loads(
        fake.generate([{"content": "Parent code:"}]).raw_text
    )
    assert "strategy" in json.loads(
        fake.generate([{"content": "Parent strategy:"}]).raw_text
    )
    assert fake.calls == 2
