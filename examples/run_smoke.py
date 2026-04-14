#!/usr/bin/env python3
"""Run a small end-to-end Stage 1 smoke test through OpenRouter.

The API key is read from the OPENROUTER_API_KEY environment variable.
No key is read from or written to any repository file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge import judge_response  # noqa: E402
from llm import LlmError, call_llm  # noqa: E402
from models import load_models, resolve  # noqa: E402
from prompt import render_prompt  # noqa: E402


DEFAULT_PROMPT_FILE = REPO_ROOT / "examples" / "example_complete_prompt.txt"
DEFAULT_PROBLEM_FILE = REPO_ROOT / "examples" / "problems_hard3_20.jsonl"
DEFAULT_MODEL_FILE = REPO_ROOT / "evaluation_models.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key: set {args.api_key_env}", file=sys.stderr)
        return 2

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    problems = _load_jsonl(args.problem_file)[: args.limit]
    configs = load_models(args.model_file)

    aliases = args.models or list(configs)
    unknown = [alias for alias in aliases if alias not in configs]
    if unknown:
        print(f"Unknown model alias(es): {', '.join(unknown)}", file=sys.stderr)
        return 2

    total = 0
    parsed = 0
    correct_count = 0
    call_timeout = args.call_timeout if args.call_timeout and args.call_timeout > 0 else None

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for alias in aliases:
            entry = configs[alias]
            provider_name, provider_config, kwargs = resolve(entry, api_keys=[api_key])
            if args.max_tokens is not None:
                kwargs.max_tokens = min(args.max_tokens, kwargs.max_tokens or args.max_tokens)

            for problem in problems:
                total += 1
                prompt_text = render_prompt(
                    prompt_template,
                    problem["equation1"],
                    problem["equation2"],
                )

                try:
                    call = call_llm(
                        client,
                        provider_name=provider_name,
                        provider_config=provider_config,
                        model_id=entry.model_id,
                        prompt=prompt_text,
                        kwargs=kwargs,
                    )
                    response = await asyncio.wait_for(call, timeout=call_timeout) if call_timeout is not None else await call
                    correct, reason = judge_response(response.text, expected_answer=problem["answer"])
                    parsed += correct is not None
                    correct_count += correct is True
                    print(
                        json.dumps(
                            {
                                "model": alias,
                                "problem_id": problem["id"],
                                "expected_answer": problem["answer"],
                                "correct": correct,
                                "reason": reason,
                                "finish_reason": response.finish_reason,
                                "tokens_in": response.tokens_in,
                                "tokens_out": response.tokens_out,
                                "actual_provider": response.actual_provider,
                                **({} if args.hide_response else {"response_text": response.text}),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except asyncio.TimeoutError:
                    print(
                        json.dumps(
                            {
                                "model": alias,
                                "problem_id": problem["id"],
                                "error": "TimeoutError",
                                "message": f"Call exceeded {call_timeout} seconds",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except LlmError as exc:
                    print(
                        json.dumps(
                            {
                                "model": alias,
                                "problem_id": problem["id"],
                                "error": type(exc).__name__,
                                "message": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    print(
        json.dumps(
            {
                "summary": {
                    "total_calls": total,
                    "parseable": parsed,
                    "correct": correct_count,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Prompt file to render; defaults to examples/example_complete_prompt.txt",
    )
    parser.add_argument("--problem-file", type=Path, default=DEFAULT_PROBLEM_FILE)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--models", nargs="*", help="Model aliases from evaluation_models.json")
    parser.add_argument("--limit", type=int, default=20, help="Number of problems to run")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional local cap for smoke tests")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=240.0,
        help="Total timeout per model/problem call in seconds; set <=0 to disable",
    )
    parser.add_argument(
        "--hide-response",
        action="store_true",
        help="Hide raw model output from per-problem JSON lines",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    return parser


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
