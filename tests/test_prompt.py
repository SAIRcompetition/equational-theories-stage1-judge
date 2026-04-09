"""Tests for prompt.py — equation placeholder substitution."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prompt import render_prompt


def test_substitutes_equations():
    t = "{{equation1}} vs {{equation2}}"
    assert render_prompt(t, "x+1", "x+1") == "x+1 vs x+1"


def test_spaced_placeholders_accepted():
    t = "{{ equation1 }} {{ equation2 }}"
    assert render_prompt(t, "A", "B") == "A B"


def test_no_placeholders_unchanged():
    t = "plain prompt text"
    assert render_prompt(t, "a", "b") == "plain prompt text"


def test_both_placeholder_forms_mixed():
    t = "{{equation1}} and {{ equation2 }}"
    assert render_prompt(t, "x^2", "(x)(x)") == "x^2 and (x)(x)"


def test_multiple_placeholders_same_equation():
    # Each placeholder is replaced once left-to-right; repeats of the same
    # equation are all substituted.
    t = "{{equation1}} equals {{equation1}}"
    assert render_prompt(t, "x+1", "ignored") == "x+1 equals x+1"
