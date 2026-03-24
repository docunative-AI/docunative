"""
Unit tests for parallel dataset writer (dataset/builder/writer.py).

Uses mocks only — no Cohere/Ollama calls.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _valid_doc_text(facts: dict) -> str:
    nums = [
        v for k, v in facts.items()
        if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    anchor = str(int(nums[0])) if nums else "1350"
    return ("word " * 120) + anchor


def test_writer_jsonl_sorted_by_domain_then_doc_idx(monkeypatch, tmp_path: Path) -> None:
    from dataset.builder import writer

    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr(writer, "_get_worker_client", lambda _ct: MagicMock())

    def fake_generate(facts, client_type, client, model_name=None):
        return _valid_doc_text(facts)

    monkeypatch.setattr(writer, "generate_document", fake_generate)

    writer.generate_all_documents(
        test_mode=True,
        language_filter="hi",
        use_ollama=False,
        max_workers=4,
    )

    out = tmp_path / "hi.jsonl"
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 4
    domains = [r["domain"] for r in rows]
    assert domains == writer.SUPPORTED_DOMAINS
    assert all(r["language"] == "hi" for r in rows)


def test_writer_respects_max_workers(monkeypatch, tmp_path: Path) -> None:
    from dataset.builder import writer

    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr(writer, "_get_worker_client", lambda _ct: MagicMock())

    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def fake_generate(facts, client_type, client, model_name=None):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.08)
        with lock:
            in_flight -= 1
        return _valid_doc_text(facts)

    monkeypatch.setattr(writer, "generate_document", fake_generate)

    writer.generate_all_documents(
        test_mode=True,
        language_filter="hi",
        use_ollama=False,
        max_workers=3,
    )

    assert peak <= 3
    assert peak >= 2


def test_generate_document_cohere_429_backoff_then_success(monkeypatch) -> None:
    import tenacity.nap

    from cohere.errors import TooManyRequestsError

    from dataset.builder import writer

    monkeypatch.setenv("COHERE_429_MAX_RETRIES", "5")
    monkeypatch.setenv("COHERE_429_INITIAL_DELAY_SEC", "0.01")
    monkeypatch.setenv("COHERE_429_MAX_DELAY_SEC", "0.02")

    sleeps: list[float] = []
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: sleeps.append(float(s)))

    client = MagicMock()
    ok_resp = MagicMock()
    ok_resp.text = ("word " * 120) + "9999"
    client.chat = MagicMock(
        side_effect=[
            TooManyRequestsError({}, headers={}),
            TooManyRequestsError({}, headers={"retry-after": "0.05"}),
            ok_resp,
        ]
    )

    facts = {"_language": "hi", "_domain": "lease", "_seed": 1, "monthly_rent": 9999}
    out = writer.generate_document(facts, "cohere", client)

    assert out == ok_resp.text
    assert client.chat.call_count == 3
    assert len(sleeps) == 2


def test_generate_document_cohere_429_exhausted(monkeypatch) -> None:
    import tenacity.nap

    from cohere.errors import TooManyRequestsError

    from dataset.builder import writer

    monkeypatch.setenv("COHERE_429_MAX_RETRIES", "2")
    monkeypatch.setenv("COHERE_429_INITIAL_DELAY_SEC", "0")
    monkeypatch.setenv("COHERE_429_MAX_DELAY_SEC", "0")

    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _s: None)

    client = MagicMock()
    client.chat = MagicMock(side_effect=TooManyRequestsError({}, headers={}))

    facts = {"_language": "pl", "_domain": "lease", "_seed": 0, "monthly_rent": 1000}
    out = writer.generate_document(facts, "cohere", client)

    assert out is None
    assert client.chat.call_count == 3


def test_resolve_max_workers_defaults(monkeypatch) -> None:
    from dataset.builder import writer

    monkeypatch.delenv("WRITER_MAX_WORKERS", raising=False)
    assert writer._resolve_max_workers("cohere", None) == writer._DEFAULT_COHERE_WORKERS
    assert writer._resolve_max_workers("ollama", None) == writer._DEFAULT_OLLAMA_WORKERS

    monkeypatch.setenv("WRITER_MAX_WORKERS", "5")
    assert writer._resolve_max_workers("cohere", None) == 5

    assert writer._resolve_max_workers("cohere", 12) == 12
    assert writer._resolve_max_workers("cohere", 0) == 1
