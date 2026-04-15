# Stage 1 Judge for the Mathematics Distillation Challenge: Equational Theories

Official local evaluation toolkit for Stage 1 of the SAIR Mathematics
Distillation Challenge: Equational Theories.

This repository lets you:

- render a complete Stage 1 prompt with `{{equation1}}` and `{{equation2}}`
- call the official evaluation models through OpenRouter with pinned providers
- extract a final `TRUE` or `FALSE` verdict
- run a small local smoke test before submission

Useful links:

- Competition overview: <https://competition.sair.foundation/competitions/mathematics-distillation-challenge-equational-theories-stage1/overview>
- Playground: <https://playground.sair.foundation/playground/mathematics-distillation-challenge-equational-theories-stage1>
- Public selected problems: <https://huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems>

Requires Python 3.9+.

---

## Quick Start

```sh
pip install -e ".[dev]"
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
python examples/run_smoke.py --limit 2 --models gpt-oss-120b
```

This command:

- loads `examples/example_complete_prompt.txt`
- loads the first 2 problems from `examples/problems_hard3_20.jsonl`
- calls `gpt-oss-120b` using `evaluation_models.json`
- grades each response with the local verdict extractor
- prints one JSON line per problem plus a final summary

`--limit` is a problem-count limit, not a token limit.
Use `--call-timeout` to keep one slow provider call from blocking the whole run.

---

## Stage 1 Prompt Format

Stage 1 uses a single complete user prompt.

Your prompt should already contain:

- your instructions
- any formatting constraints
- any cheatsheet content you want to include

Supported placeholders:

- `{{equation1}}` or `{{ equation1 }}`
- `{{equation2}}` or `{{ equation2 }}`

`prompt.py` only substitutes these placeholders. It does not run a full
template engine.

For Stage 1, format stability matters. A strong prompt still loses if the final
verdict is not parseable.

---

## Official Evaluation Setup

The official local model config is `evaluation_models.json`.

All three official routes use:

- strict provider pinning with `allow_fallbacks = false`
- temperature `0.0`
- max output tokens `8192`
- seeded requests where supported

Current planned setup:

| Alias | OpenRouter model | Pinned provider route | Reasoning | Seed |
|-------|------------------|-----------------------|-----------|------|
| `gpt-oss-120b` | `openai/gpt-oss-120b` | `deepinfra/bf16` | low | `0` |
| `llama-3-3-70b-instruct` | `meta-llama/llama-3.3-70b-instruct` | `deepinfra/fp8` | disabled | `0` |
| `gemma-4-31b-it` | `google/gemma-4-31b-it` | `novita/bf16` | disabled | `0` |

These three models are currently planned to serve as the final Stage 1
evaluation models, with equal weight. The setup may still be adjusted based on
community feedback.

In local smoke tests, you may pass `--max-tokens` to temporarily lower the cap
for speed or cost. The default local testing setup in this repository remains
`8192`.

Organizers also plan to run an additional `16384`-token evaluation and display
those results in a separate leaderboard. The primary Stage 1 leaderboard
remains based on the `8192` setting in this repository.

If a provider route for one of the planned evaluation models remains unstable
by the end of Stage 1, organizers may use a SAIR-hosted serving setup for that
same model and fixed configuration in final evaluation, while ensuring fairness
of the evaluation.

---

## Local Workflow

### 1. Write or edit a prompt file

You can start from:

- `examples/example_complete_prompt.txt`

Or pass your own prompt file:

```sh
python examples/run_smoke.py --prompt-file /path/to/your_prompt.txt --limit 20 --models gpt-oss-120b
```

### 2. Run a smoke test

Example:

```sh
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
python examples/run_smoke.py --limit 20 --models gpt-oss-120b llama-3-3-70b-instruct
```

Example output:

```json
{"model":"gpt-oss-120b","problem_id":"hard3_0001","expected_answer":true,"correct":true,"reason":"Answered TRUE","finish_reason":"stop","tokens_in":174,"tokens_out":697,"actual_provider":"DeepInfra","response_text":"VERDICT: TRUE\nREASONING: ..."}
{"summary":{"total_calls":20,"parseable":20,"correct":9}}
```

The runner prints the raw model output in `response_text` by default. Use
`--hide-response` if you want a lighter per-problem JSON format.

### 3. Check the important fields

- `correct`: whether the model matched the ground truth
- `reason`: how the judge interpreted the answer
- `finish_reason`: useful for spotting truncation such as `length`
- `actual_provider`: the provider OpenRouter actually used

Recommended loop:

1. edit your prompt
2. run local smoke tests on public problems
3. fix formatting failures first
4. then optimize accuracy
5. submit only after the prompt is stable locally

---

## Example Assets

The `examples/` folder contains:

- `examples/example_complete_prompt.txt`
- `examples/problems_hard3_20.jsonl`
- `examples/run_smoke.py`

`problems_hard3_20.jsonl` contains 20 public example problems from the `hard3`
subset: the first 10 TRUE problems and the first 10 FALSE problems, preserving
the original order.

---

## Repository Layout

| File | Purpose |
|------|---------|
| `prompt.py` | fills equation placeholders into a complete prompt |
| `judge.py` | extracts and grades `TRUE` / `FALSE` verdicts |
| `llm.py` | OpenRouter client with retry, provider routing, and response parsing |
| `models.py` | loads `evaluation_models.json` and resolves call parameters |
| `evaluation_models.json` | official local model configuration |
| `examples/` | prompt example, public smoke-test problems, and runner |
| `tests/` | automated tests for prompt rendering, verdict extraction, model config, and smoke-runner behavior |

---

## Developer Notes

Important verdict-extraction behavior:

- matching is case-insensitive
- boxed answers beat labeled answers
- labeled answers beat bare first/last-line answers
- within the same marker type, the last occurrence wins
- instruction patterns such as `VERDICT: TRUE or FALSE` are ignored

Run the test suite:

```sh
pytest tests/
```

`llm.py` logs through Python's standard `logging` module under the logger name
`llm`.
