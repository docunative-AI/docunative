"""
Unit tests for parallel LLM QA factory (dataset/builder/qa_factory_llm.py).

Mocks only — no Cohere calls.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _minimal_doc(doc_id: str, language: str, domain: str) -> dict:
    return {
        "doc_id": doc_id,
        "language": language,
        "domain": domain,
        "document_text": "placeholder " * 200,
        "facts": {"monthly_rent": 1350},
    }


def test_qa_factory_llm_sorted_by_language_domain_doc_idx(monkeypatch, tmp_path: Path) -> None:
    from dataset.builder import qa_factory_llm as m

    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr(m, "_get_thread_cohere_client", lambda: MagicMock())

    docs = [
        _minimal_doc("hi_immigration_letter_0", "hi", "immigration_letter"),
        _minimal_doc("hi_lease_0", "hi", "lease"),
        _minimal_doc("hi_employment_0", "hi", "employment"),
        _minimal_doc("hi_health_insurance_0", "hi", "health_insurance"),
    ]

    def fake_generate(doc, client, n_pairs=10, model="c4ai-aya-expanse-32b"):
        return [
            {
                "doc_id": doc["doc_id"],
                "language": doc["language"],
                "domain": doc["domain"],
                "question": doc["doc_id"],
                "answer": "ok",
                "field": "llm_generated",
            }
        ]

    monkeypatch.setattr(m, "generate_llm_qa_pairs", fake_generate)

    (tmp_path / "hi.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "qa_pairs_llm.jsonl"
    m.generate_all_llm_qa_pairs(
        languages=["hi"],
        domains=None,
        test_mode=False,
        docs_dir=tmp_path,
        output_path=out,
        max_workers=4,
    )

    assert out.exists()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 4
    questions = [r["question"] for r in rows]
    assert questions == [
        "hi_lease_0",
        "hi_employment_0",
        "hi_health_insurance_0",
        "hi_immigration_letter_0",
    ]


def test_qa_factory_llm_respects_max_workers(monkeypatch, tmp_path: Path) -> None:
    from dataset.builder import qa_factory_llm as m

    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr(m, "_get_thread_cohere_client", lambda: MagicMock())

    docs = [_minimal_doc(f"hi_lease_{i}", "hi", "lease") for i in range(4)]
    (tmp_path / "hi.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n",
        encoding="utf-8",
    )

    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def fake_generate(doc, client, n_pairs=10, model="c4ai-aya-expanse-32b"):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.08)
        with lock:
            in_flight -= 1
        return [
            {
                "doc_id": doc["doc_id"],
                "language": doc["language"],
                "domain": doc["domain"],
                "question": "q",
                "answer": "a",
                "field": "llm_generated",
            }
        ]

    monkeypatch.setattr(m, "generate_llm_qa_pairs", fake_generate)
    out = tmp_path / "qa.jsonl"
    m.generate_all_llm_qa_pairs(
        languages=["hi"],
        docs_dir=tmp_path,
        output_path=out,
        max_workers=3,
    )

    assert peak <= 3
    assert peak >= 2


def test_resolve_max_workers_qa_factory(monkeypatch) -> None:
    from dataset.builder import qa_factory_llm as m

    monkeypatch.delenv("QA_FACTORY_LLM_MAX_WORKERS", raising=False)
    assert m._resolve_max_workers(None) == m._DEFAULT_QA_FACTORY_WORKERS

    monkeypatch.setenv("QA_FACTORY_LLM_MAX_WORKERS", "5")
    assert m._resolve_max_workers(None) == 5

    assert m._resolve_max_workers(12) == 12
    assert m._resolve_max_workers(0) == 1


def test_generate_llm_qa_cohere_429_backoff_then_success(monkeypatch) -> None:
    import tenacity.nap

    from cohere.errors import TooManyRequestsError

    from dataset.builder import qa_factory_llm as m

    monkeypatch.setenv("COHERE_429_MAX_RETRIES", "5")
    monkeypatch.setenv("COHERE_429_INITIAL_DELAY_SEC", "0.01")
    monkeypatch.setenv("COHERE_429_MAX_DELAY_SEC", "0.02")

    sleeps: list[float] = []
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: sleeps.append(float(s)))

    ok_json = '[{"question": "What is rent?", "answer": "Monthly rent is 1350 EUR"}]'
    ok_resp = MagicMock()
    ok_resp.text = ok_json

    client = MagicMock()
    client.chat = MagicMock(
        side_effect=[
            TooManyRequestsError({}, headers={}),
            TooManyRequestsError({}, headers={"retry-after": "0.05"}),
            ok_resp,
        ]
    )

    doc = {
        "doc_id": "hi_lease_0",
        "language": "hi",
        "domain": "lease",
        "document_text": "x " * 500,
        "facts": {"monthly_rent": 1350},
    }
    pairs = m.generate_llm_qa_pairs(doc, client, n_pairs=1, model="c4ai-aya-expanse-32b")

    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is rent?"
    assert client.chat.call_count == 3
    assert len(sleeps) == 2


def test_generate_llm_qa_cohere_429_exhausted(monkeypatch) -> None:
    import tenacity.nap

    from cohere.errors import TooManyRequestsError

    from dataset.builder import qa_factory_llm as m

    monkeypatch.setenv("COHERE_429_MAX_RETRIES", "2")
    monkeypatch.setenv("COHERE_429_INITIAL_DELAY_SEC", "0")
    monkeypatch.setenv("COHERE_429_MAX_DELAY_SEC", "0")

    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _s: None)

    client = MagicMock()
    client.chat = MagicMock(side_effect=TooManyRequestsError({}, headers={}))

    doc = {
        "doc_id": "pl_lease_0",
        "language": "pl",
        "domain": "lease",
        "document_text": "y " * 500,
        "facts": {"monthly_rent": 1000},
    }
    pairs = m.generate_llm_qa_pairs(doc, client, n_pairs=1)

    assert pairs == []
    # Each outer attempt (MAX_RETRIES+1=3) calls cohere_chat; each exhausts 3 inner 429 attempts.
    assert client.chat.call_count == 9
