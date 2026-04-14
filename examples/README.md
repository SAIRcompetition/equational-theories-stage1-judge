# Prompt and problem examples

Stage 1 sends one complete user prompt to the model. The prompt renderer in
this repository substitutes only `{{equation1}}` / `{{ equation1 }}` and
`{{equation2}}` / `{{ equation2 }}`.

## Files

- `example_complete_prompt.txt` is an example complete prompt. You can edit it
  directly for quick experiments, or pass your own prompt file with
  `--prompt-file`.
- `problems_hard3_20.jsonl` contains 20 public example problems selected from
  the `hard3` subset of `SAIRfoundation/equational-theories-selected-problems`
  (https://huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems):
  the first 10 TRUE problems and the first 10 FALSE problems, preserving the
  original `hard3` order.
- `run_smoke.py` runs the prompt and problems through the configured
  OpenRouter models, prints the raw model output by default, and grades each
  response.

## Run

```sh
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
python examples/run_smoke.py --limit 2 --models gpt-oss-120b
```

`--limit 2` means: run only the first 2 problems from
`problems_hard3_20.jsonl`. It is a problem-count limit, not a token limit.

If a provider is slow, you can cap total wall-clock time per call:

```sh
python examples/run_smoke.py --limit 2 --models gpt-oss-120b --call-timeout 60
```

If you want a lighter output format, you can hide the raw model response:

```sh
python examples/run_smoke.py --limit 2 --models gpt-oss-120b --hide-response
```
