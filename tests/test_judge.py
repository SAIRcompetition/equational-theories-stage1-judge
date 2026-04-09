"""Tests for judge.py — verdict extraction and correctness grading."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from judge import judge_response


def test_verdict_true():
    correct, _ = judge_response("VERDICT: TRUE", True)
    assert correct is True


def test_verdict_false():
    correct, _ = judge_response("VERDICT: FALSE", False)
    assert correct is True


def test_verdict_wrong():
    correct, _ = judge_response("VERDICT: TRUE", False)
    assert correct is False


def test_verdict_case_insensitive():
    correct, _ = judge_response("verdict: true", True)
    assert correct is True


def test_boxed_verdict_true():
    correct, reason = judge_response(r"\boxed{TRUE}", True)
    assert correct is True
    assert reason == "Answered TRUE"


def test_boxed_verdict_with_double_backslash():
    correct, reason = judge_response(r"\\boxed{FALSE}", False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_boxed_verdict_with_markdown_value():
    correct, reason = judge_response(r"\boxed{**FALSE**}", False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_boxed_verdict_with_latex_wrapper():
    correct, reason = judge_response(r"\boxed{\text{TRUE}}", True)
    assert correct is True
    assert reason == "Answered TRUE"


def test_boxed_placeholder_answer_is_ignored():
    response = r"Format reminder: \boxed{answer}"
    correct, reason = judge_response(response, True)
    assert correct is None
    assert "No VERDICT found" in reason


def test_no_verdict():
    correct, reason = judge_response("I think the answer is yes", True)
    assert correct is None
    assert "No VERDICT found" in reason


def test_json_with_verdict_still_parsed():
    response = r'{"verdict": "TRUE", "reason": "because..."} VERDICT: TRUE'
    correct, _ = judge_response(response, True)
    assert correct is True


def test_json_only_no_verdict_marker():
    # Without a VERDICT: marker, the regex won't find it
    response = r'{"verdict": "FALSE", "reason": "nope"}'
    correct, _ = judge_response(response, False)
    assert correct is None


def test_fallback_template_uses_regex():
    correct, _ = judge_response("VERDICT: TRUE", True)
    assert correct is True


def test_bold_false_on_first_line_is_parsed():
    # **FALSE** → strip markdown → "FALSE" on the first line → leading-line detection
    response = "**FALSE**\n\nSome reasoning here..."
    correct, reason = judge_response(response, False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_standalone_without_verdict_prefix_is_unparsed():
    response = "Here is my analysis:\n\nFALSE\n\nReasoning..."
    correct, reason = judge_response(response, False)
    assert correct is None
    assert "No VERDICT found" in reason


def test_no_false_positive_in_text():
    response = "The statement is FALSE because of reasons, but I need more info."
    correct, reason = judge_response(response, False)
    assert correct is None
    assert "No VERDICT found" in reason


def test_full_ignores_instruction_style_verdict_line():
    response = "Output format:\nVERDICT: TRUE or FALSE\nREASONING:\n...\nVERDICT: FALSE"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_conflicting_verdicts_use_last_labeled_marker():
    response = "VERDICT: TRUE\nREASONING: draft\nVERDICT: FALSE"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert "conflict resolved by final labeled marker" in reason


def test_conflicting_verdict_between_verdict_and_boxed_prefers_boxed():
    response = r"VERDICT: TRUE" + "\nFinal: \\boxed{FALSE}"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert "conflict resolved by final boxed marker" in reason


def test_multiple_same_verdicts_treated_as_one():
    response = "VERDICT: FALSE\nREASONING: draft\nVERDICT: FALSE"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_bold_verdict_line():
    response = "**VERDICT: TRUE**\n\nBecause the equations are equivalent."
    correct, reason = judge_response(response, True)
    assert correct is True
    assert "no VERDICT prefix" not in reason


def test_bold_verdict_keyword_only():
    response = "**VERDICT**: FALSE\n\nNot equivalent."
    correct, _ = judge_response(response, False)
    assert correct is True


def test_bold_value_only():
    response = "VERDICT: **TRUE**"
    correct, _ = judge_response(response, True)
    assert correct is True


def test_bold_italic_verdict():
    response = "***VERDICT: FALSE***"
    correct, _ = judge_response(response, False)
    assert correct is True


def test_backtick_verdict():
    response = "VERDICT: `TRUE`"
    correct, _ = judge_response(response, True)
    assert correct is True


def test_underscore_bold_verdict():
    response = "__VERDICT: TRUE__"
    correct, _ = judge_response(response, True)
    assert correct is True


def test_trailing_final_answer_line_is_parsed():
    response = "Reasoning...\nFinal answer: FALSE"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert reason == "Answered FALSE"


# ---------------------------------------------------------------------------
# New: \text{} labeled pattern
# ---------------------------------------------------------------------------

def test_latex_text_true():
    # Standalone \text{TRUE} is recognised as a labeled verdict.
    correct, reason = judge_response(r"\text{TRUE}", True)
    assert correct is True
    assert reason == "Answered TRUE"


def test_latex_text_false():
    correct, reason = judge_response(r"\text{FALSE}", False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_latex_text_in_display_math_block():
    # \[\n\text{TRUE}\n\] — the format Jiaxuan Zou's cheatsheet instructs.
    response = "The expressions are equivalent.\n\\[\n\\text{TRUE}\n\\]"
    correct, reason = judge_response(response, True)
    assert correct is True
    assert reason == "Answered TRUE"


def test_latex_text_case_insensitive():
    correct, _ = judge_response(r"\text{true}", True)
    assert correct is True


def test_latex_text_with_spaces():
    correct, _ = judge_response(r"\text{ FALSE }", False)
    assert correct is True


# ---------------------------------------------------------------------------
# New: leading-line detection
# ---------------------------------------------------------------------------

def test_leading_line_bare_true():
    # Model answers on the first line then explains below.
    response = "TRUE\n\nBecause both sides simplify to x^2."
    correct, reason = judge_response(response, True)
    assert correct is True
    assert reason == "Answered TRUE"


def test_leading_line_bold_false():
    # **FALSE** on first line, stripped to bare FALSE by markdown cleaning.
    response = "**FALSE**\n\nThe magma operation tables differ."
    correct, reason = judge_response(response, False)
    assert correct is True
    assert reason == "Answered FALSE"


def test_leading_line_overridden_by_labeled():
    # Explicit VERDICT: keyword (priority 2) overrides bare leading line (priority 1).
    # The conflicting markers are noted in the reason string.
    response = "FALSE\n\nAfter further analysis...\nVERDICT: TRUE"
    correct, reason = judge_response(response, True)
    assert correct is True
    assert "Answered TRUE" in reason
    assert "conflict resolved by final labeled marker" in reason


def test_leading_line_conflict_trailing_wins():
    # When first line says TRUE and last line says FALSE, trailing-line wins
    # (higher index within the same priority level).
    response = "TRUE\n\nOn reflection, the answer is:\nFALSE"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert "conflict resolved by final line marker" in reason


# ---------------------------------------------------------------------------
# Slash-separated instruction patterns (TRUE/FALSE) must be unparsed
# ---------------------------------------------------------------------------

def test_verdict_slash_pattern_is_unparsed():
    # "VERDICT: TRUE/FALSE" is an instruction format hint, not a real answer.
    correct, reason = judge_response("Reply with VERDICT: TRUE/FALSE", False)
    assert correct is None
    assert "No VERDICT found" in reason


def test_answer_slash_pattern_is_unparsed():
    correct, reason = judge_response("ANSWER: TRUE/FALSE", True)
    assert correct is None
    assert "No VERDICT found" in reason


def test_final_answer_slash_pattern_is_unparsed():
    correct, reason = judge_response("FINAL ANSWER: TRUE/FALSE", True)
    assert correct is None
    assert "No VERDICT found" in reason


def test_result_slash_pattern_is_unparsed():
    correct, reason = judge_response("RESULT: FALSE/TRUE", False)
    assert correct is None
    assert "No VERDICT found" in reason


def test_slash_pattern_with_real_answer_after():
    # Instruction hint early, real answer later — the real answer wins.
    response = "Format: VERDICT: TRUE/FALSE\n\nVERDICT: FALSE"
    correct, reason = judge_response(response, False)
    assert correct is True
    assert reason == "Answered FALSE"
