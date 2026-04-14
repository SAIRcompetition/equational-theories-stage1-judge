"""Tests for examples/run_smoke.py."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

import run_smoke


@pytest.mark.asyncio
async def test_run_smoke_reports_timeout_and_continues(monkeypatch, capsys, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Q1={{equation1}} Q2={{equation2}}", encoding="utf-8")

    problem_file = tmp_path / "problems.jsonl"
    rows = [
        {"id": "p1", "equation1": "a", "equation2": "b", "answer": True},
        {"id": "p2", "equation1": "c", "equation2": "d", "answer": False},
    ]
    with problem_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    class DummyEntry:
        model_id = "openai/gpt-oss-120b"

    class DummyKwargs:
        max_tokens = 128

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(run_smoke, "load_models", lambda path: {"demo": DummyEntry()})
    monkeypatch.setattr(
        run_smoke,
        "resolve",
        lambda entry, api_keys: ("demo-provider", object(), DummyKwargs()),
    )

    calls = {"count": 0}

    async def fake_call_llm(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            await asyncio.sleep(0.05)
        return type(
            "Resp",
            (),
            {
                "text": "VERDICT: FALSE",
                "finish_reason": "stop",
                "tokens_in": 1,
                "tokens_out": 1,
                "actual_provider": "DeepInfra",
                "refusal": None,
            },
        )()

    monkeypatch.setattr(run_smoke, "call_llm", fake_call_llm)

    args = argparse.Namespace(
        prompt_file=prompt_file,
        problem_file=problem_file,
        model_file=tmp_path / "models.json",
        models=["demo"],
        limit=2,
        max_tokens=None,
        timeout=180,
        call_timeout=0.01,
        hide_response=False,
        api_key_env="OPENROUTER_API_KEY",
    )

    rc = await run_smoke._run(args)
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["error"] == "TimeoutError"
    assert lines[0]["problem_id"] == "p1"
    assert lines[1]["problem_id"] == "p2"
    assert lines[1]["correct"] is True
    assert lines[1]["response_text"] == "VERDICT: FALSE"
    assert lines[2]["summary"] == {"total_calls": 2, "parseable": 1, "correct": 1}


@pytest.mark.asyncio
async def test_run_smoke_hides_response_when_requested(monkeypatch, capsys, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Q1={{equation1}} Q2={{equation2}}", encoding="utf-8")

    problem_file = tmp_path / "problems.jsonl"
    row = {"id": "p1", "equation1": "a", "equation2": "b", "answer": True}
    problem_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class DummyEntry:
        model_id = "openai/gpt-oss-120b"

    class DummyKwargs:
        max_tokens = 128

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(run_smoke, "load_models", lambda path: {"demo": DummyEntry()})
    monkeypatch.setattr(
        run_smoke,
        "resolve",
        lambda entry, api_keys: ("demo-provider", object(), DummyKwargs()),
    )

    async def fake_call_llm(*args, **kwargs):
        return type(
            "Resp",
            (),
            {
                "text": "VERDICT: TRUE",
                "finish_reason": "stop",
                "tokens_in": 1,
                "tokens_out": 1,
                "actual_provider": "DeepInfra",
                "refusal": None,
            },
        )()

    monkeypatch.setattr(run_smoke, "call_llm", fake_call_llm)

    args = argparse.Namespace(
        prompt_file=prompt_file,
        problem_file=problem_file,
        model_file=tmp_path / "models.json",
        models=["demo"],
        limit=1,
        max_tokens=None,
        timeout=180,
        call_timeout=0.01,
        hide_response=True,
        api_key_env="OPENROUTER_API_KEY",
    )

    rc = await run_smoke._run(args)
    assert rc == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["problem_id"] == "p1"
    assert "response_text" not in lines[0]
    assert lines[1]["summary"] == {"total_calls": 1, "parseable": 1, "correct": 1}
