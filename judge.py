"""
TRUE/FALSE verdict extraction for the Mathematics Distillation Challenge:
Equational Theories, Stage 1.

Given a model's free-form text response and the ground-truth boolean answer,
:func:`judge_response` locates a ``TRUE`` or ``FALSE`` verdict and checks it
against the expected value.

Three marker types are recognised, in descending priority:

1. **Boxed** (priority 3) — LaTeX ``\\boxed{TRUE}`` / ``\\boxed{FALSE}``.
   Nested braces, Markdown bold/italic, and LaTeX wrappers such as
   ``\\text{}``, ``\\mathrm{}``, ``\\mathbf{}``, ``\\operatorname{}`` are all
   handled transparently.

2. **Labeled** (priority 2) — a keyword followed by the verdict::

       VERDICT: TRUE          ANSWER: FALSE
       FINAL ANSWER: TRUE     RESULT: FALSE
       OUTPUT_RESULT: TRUE    \text{TRUE}

   Both ASCII ``:`` and full-width ``：`` are accepted as separators, as are
   ``=`` and ``-`` for ANSWER / FINAL ANSWER / RESULT / OUTPUT_RESULT.
   ``\\text{TRUE/FALSE}`` (LaTeX text command, e.g. inside ``\\[…\\]``
   display-math blocks) is also recognised.

3. **Line** (priority 1) — the **first or last** non-empty line of the
   response consists solely of ``TRUE`` or ``FALSE`` (with an optional
   ``FINAL ANSWER:`` prefix and optional trailing punctuation).  When both
   the first and last line fire, the trailing (last) line wins via the
   rightmost-occurrence rule.

Tie-breaking rules
------------------
* The **highest-priority type** wins over lower-priority types regardless of
  position.
* Within the same type, the **last occurrence** in the text wins.  This
  means instruction-style preambles in the prompt — such as
  *"Reply with VERDICT: TRUE or FALSE"* — are automatically overridden by
  the model's actual answer that appears later.

When conflicting markers are resolved, the ``reason`` return value notes
which marker was chosen.

Edge cases handled
------------------
* Markdown wrappers (``***``, ``**``, ``__``, `` ` ``) stripped before
  matching, so ``**VERDICT: TRUE**`` is parsed correctly.
* ``\\boxed{answer}`` (the literal placeholder word *answer*) is ignored.
* ``VERDICT: TRUE or FALSE`` — the "or" clause marks this as an instruction
  pattern, not a real verdict, and is skipped.

Examples
--------
Correct answer via ``\\boxed{}``:

>>> from judge import judge_response
>>> judge_response(r"\\boxed{TRUE}", expected_answer=True)
(True, 'Answered TRUE')

Wrong answer via labeled marker:

>>> judge_response("VERDICT: FALSE", expected_answer=True)
(False, 'Answered FALSE, expected TRUE')

No parseable verdict:

>>> judge_response("I believe Equation 1 implies Equation 2.", expected_answer=True)
(None, 'No VERDICT found in response')
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["judge_response"]


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level, compiled once)
# ---------------------------------------------------------------------------

# Matches one or more backslashes followed by \boxed{
_BOXED_START_RE = re.compile(r"(?i)\\+boxed\s*\{")

# VERDICT: TRUE  /  VERDICT: FALSE  (ASCII or full-width colon)
_VERDICT_RE = re.compile(r"(?i)\bVERDICT\s*[:：]\s*(TRUE|FALSE)\b")

# ANSWER / FINAL ANSWER / RESULT / OUTPUT_RESULT  followed by := or -
_ANSWER_RE = re.compile(
    r"(?i)\b(?:FINAL\s+ANSWER|ANSWER|OUTPUT_RESULT|RESULT)\s*[:：=-]\s*(TRUE|FALSE)\b"
)

# \text{TRUE} or \text{FALSE}  — LaTeX text command used in display-math blocks
_LATEX_TEXT_RE = re.compile(r"(?i)\\text\s*\{\s*(TRUE|FALSE)\s*\}")

# First / last line: optional "FINAL ANSWER:" prefix, then TRUE or FALSE, trailing punctuation OK
_LINE_RE = re.compile(
    r"(?i)^\s*(?:FINAL\s+ANSWER\s*[:：=-]\s*)?(TRUE|FALSE)\s*[.!?]*\s*$"
)

# Unwrap \text{…}, \mathrm{…}, \mathbf{…}, \operatorname{…}
_LATEX_WRAPPER_RE = re.compile(
    r"(?is)^\\(?:text|mathrm|mathbf|operatorname)\s*\{(.+)\}$"
)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

class _Source(Enum):
    """Verdict source, with numeric priority (higher = wins)."""
    LINE    = 1   # first or last non-empty line (bare TRUE / FALSE)
    LABELED = 2   # keyword marker: VERDICT:, ANSWER:, \text{}, etc.
    BOXED   = 3   # \boxed{TRUE} / \boxed{FALSE}

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class _Candidate:
    value:  bool     # True → model said TRUE; False → model said FALSE
    source: _Source
    index:  int      # byte offset of match start (0 for leading-line, len(response) for trailing)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(s: str) -> str:
    """Remove bold / italic / code Markdown wrappers before regex matching."""
    # Multi-char sequences must be removed before single-char ones.
    return s.replace("***", "").replace("**", "").replace("__", "").replace("`", "")


def _parse_bool(label: str) -> Optional[bool]:
    """Return True/False for 'TRUE'/'FALSE' (case-insensitive), else None."""
    u = label.upper()
    if u == "TRUE":
        return True
    if u == "FALSE":
        return False
    return None


def _is_or_clause(response: str, match_end: int) -> bool:
    """Return True if the text immediately after *match_end* is an instruction-pattern suffix.

    Handles two forms:
    - ``VERDICT: TRUE or FALSE``  — "or" alternative
    - ``VERDICT: TRUE/FALSE``     — slash-separated alternative
    """
    after = response[match_end:].split("\n", 1)[0].lstrip()
    if after[:2].upper() == "OR":
        rest = after[2:]
        return not rest or rest[0].isspace()
    if after.startswith("/"):
        return bool(re.match(r"(?i)(TRUE|FALSE)\b", after[1:].lstrip()))
    return False


def _parse_boxed_content(token: str) -> Optional[bool]:
    """Strip punctuation and LaTeX wrappers from ``\\boxed{…}`` content, then parse.

    Returns None for the placeholder literal ``answer``.
    """
    STRIP = " \t\r\n.,;:!?$()[]"
    current = token.strip()
    for _ in range(4):   # at most 4 unwrap passes (e.g. \text{\mathbf{TRUE}})
        current = current.strip(STRIP)
        if current.upper() == "ANSWER":
            return None          # \boxed{answer} is a placeholder, not a verdict
        verdict = _parse_bool(current)
        if verdict is not None:
            return verdict
        m = _LATEX_WRAPPER_RE.match(current)
        if m:
            current = m.group(1)
            continue
        break
    return None


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _extract_boxed(response: str, out: list[_Candidate]) -> None:
    """Find all \\boxed{…} expressions and collect their verdicts."""
    for m in _BOXED_START_RE.finditer(response):
        # Manually track brace depth to handle nested braces.
        depth = 1
        content_start = m.end()
        content_end = None
        for i, ch in enumerate(response[content_start:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    content_end = content_start + i
                    break
        if content_end is None:
            continue
        value = _parse_boxed_content(response[content_start:content_end])
        if value is not None:
            out.append(_Candidate(value=value, source=_Source.BOXED, index=m.start()))


def _extract_labeled(response: str, out: list[_Candidate]) -> None:
    """Find all labeled markers (VERDICT:, ANSWER:, \\text{}, …) and collect verdicts."""
    for pattern in (_VERDICT_RE, _ANSWER_RE, _LATEX_TEXT_RE):
        for m in pattern.finditer(response):
            if _is_or_clause(response, m.end()):
                continue    # skip instruction patterns like "VERDICT: TRUE or FALSE"
            value = _parse_bool(m.group(1))
            if value is not None:
                out.append(_Candidate(value=value, source=_Source.LABELED, index=m.start()))


def _extract_leading_line(response: str, out: list[_Candidate]) -> None:
    """Check whether the first non-empty line is a bare TRUE/FALSE verdict."""
    first = next(
        (line for line in response.splitlines() if line.strip()),
        None,
    )
    if first is None:
        return
    m = _LINE_RE.match(first)
    if not m:
        return
    value = _parse_bool(m.group(1))
    if value is not None:
        # Index 0: leading-line always loses to trailing-line in same-priority tie-break.
        out.append(_Candidate(value=value, source=_Source.LINE, index=0))


def _extract_trailing_line(response: str, out: list[_Candidate]) -> None:
    """Check whether the last non-empty line is a bare TRUE/FALSE verdict."""
    last = next(
        (line for line in reversed(response.splitlines()) if line.strip()),
        None,
    )
    if last is None:
        return
    m = _LINE_RE.match(last)
    if not m:
        return
    value = _parse_bool(m.group(1))
    if value is not None:
        # len(response) ensures trailing-line always beats leading-line in ties.
        out.append(_Candidate(value=value, source=_Source.LINE, index=len(response)))


def _best_candidate(candidates: list[_Candidate]) -> Optional[_Candidate]:
    """Return the winning candidate: highest priority, then rightmost position."""
    if not candidates:
        return None
    top_priority = max(c.source.value for c in candidates)
    top = [c for c in candidates if c.source.value == top_priority]
    return max(top, key=lambda c: c.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_response(response: str, expected_answer: bool) -> tuple[Optional[bool], str]:
    """Judge a model response against the expected TRUE/FALSE answer.

    Parameters
    ----------
    response:
        The model's raw text output.
    expected_answer:
        Ground-truth implication label (``True`` = Equation 1 implies
        Equation 2 over all magmas; ``False`` = it does not).

    Returns
    -------
    correct : bool or None
        ``True``  — a verdict was found and it matches *expected_answer*.
        ``False`` — a verdict was found but it contradicts *expected_answer*.
        ``None``  — no parseable verdict marker was found in *response*.
    reason : str
        A short human-readable explanation of the outcome, suitable for
        logging or storage alongside the result.

    Examples
    --------
    Correct boxed answer:

    >>> judge_response(r"\\boxed{TRUE}", expected_answer=True)
    (True, 'Answered TRUE')

    Wrong labeled answer:

    >>> judge_response("VERDICT: FALSE", expected_answer=True)
    (False, 'Answered FALSE, expected TRUE')

    No verdict found:

    >>> judge_response("The two expressions look the same to me.", expected_answer=True)
    (None, 'No VERDICT found in response')

    Instruction preamble overridden by final answer:

    >>> judge_response(
    ...     "Output format: VERDICT: TRUE or FALSE\\nVERDICT: FALSE",
    ...     expected_answer=False,
    ... )
    (True, 'Answered FALSE')
    """
    cleaned = _strip_markdown(response)

    candidates: list[_Candidate] = []
    _extract_boxed(cleaned, candidates)
    _extract_labeled(cleaned, candidates)
    _extract_leading_line(cleaned, candidates)
    _extract_trailing_line(cleaned, candidates)

    chosen = _best_candidate(candidates)
    if chosen is None:
        return None, "No VERDICT found in response"

    verdict = chosen.value
    correct = verdict == expected_answer
    ver_label = "TRUE" if verdict else "FALSE"
    exp_label = "TRUE" if expected_answer else "FALSE"

    reason = f"Answered {ver_label}" if correct else f"Answered {ver_label}, expected {exp_label}"

    # Note conflicts so callers can inspect ambiguous responses.
    if any(c.value for c in candidates) and any(not c.value for c in candidates):
        reason += f" (conflict resolved by final {chosen.source.label} marker)"

    return correct, reason


# ---------------------------------------------------------------------------
# CLI — useful for quick manual testing of prompts
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Judge a model response for a TRUE/FALSE verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python judge.py "VERDICT: TRUE" --expected true\n'
            '  echo "\\\\boxed{FALSE}" | python judge.py --expected false\n'
            '  python judge.py "VERDICT: TRUE" --expected true --json'
        ),
    )
    parser.add_argument(
        "response",
        nargs="?",
        help="Model response text.  Omit to read from stdin.",
    )
    parser.add_argument(
        "--expected",
        required=True,
        choices=["true", "false", "True", "False", "TRUE", "FALSE"],
        metavar="true|false",
        help="Ground-truth answer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON object instead of plain text.",
    )
    args = parser.parse_args()

    text     = args.response if args.response is not None else sys.stdin.read()
    expected = args.expected.upper() == "TRUE"

    correct, reason = judge_response(text, expected)

    if args.json:
        print(json.dumps({"correct": correct, "reason": reason}))
    else:
        status = (
            "CORRECT"    if correct is True  else
            "WRONG"      if correct is False else
            "NO_VERDICT"
        )
        print(f"{status}: {reason}")


if __name__ == "__main__":
    _main()
