"""
Prompt template rendering for the Mathematics Distillation Challenge:
Equational Theories, Stage 1.

Your cheatsheet IS the full prompt that gets sent to the model.  Drop
``{{equation1}}`` and ``{{equation2}}`` anywhere in the text and
:func:`render_prompt` will substitute the actual equation strings before
the prompt is dispatched.

Example
-------
>>> from prompt import render_prompt
>>> template = "Does {{equation1}} imply {{equation2}}? Reply VERDICT: TRUE or FALSE."
>>> render_prompt(template, "x^2 - 1", "(x-1)(x+1)")
'Does x^2 - 1 imply (x-1)(x+1)? Reply VERDICT: TRUE or FALSE.'
"""

from __future__ import annotations

__all__ = ["render_prompt"]


def render_prompt(prompt_text: str, equation1: str, equation2: str) -> str:
    """Substitute equation variables into a cheatsheet template.

    Parameters
    ----------
    prompt_text:
        The cheatsheet content, which becomes the full prompt sent to the
        model.
    equation1:
        First equation string (replaces ``{{equation1}}``).
    equation2:
        Second equation string (replaces ``{{equation2}}``).

    Returns
    -------
    str
        The rendered prompt, ready to be passed to :func:`~llm.call_llm`.

    Notes
    -----
    Both ``{{var}}`` and ``{{ var }}`` (with inner spaces) are accepted.
    Substitutions are applied left-to-right with no recursive expansion —
    placeholders within an equation value are left as-is.  Equations are
    inserted verbatim with no escaping.

    Placeholder reference
    ~~~~~~~~~~~~~~~~~~~~~
    ============================  =======================
    Placeholder                    Replaced with
    ============================  =======================
    ``{{equation1}}``              First equation string
    ``{{ equation1 }}``            First equation string
    ``{{equation2}}``              Second equation string
    ``{{ equation2 }}``            Second equation string
    ============================  =======================
    """
    prompt = prompt_text
    prompt = prompt.replace("{{equation1}}", equation1)
    prompt = prompt.replace("{{ equation1 }}", equation1)
    prompt = prompt.replace("{{equation2}}", equation2)
    prompt = prompt.replace("{{ equation2 }}", equation2)
    return prompt


# ---------------------------------------------------------------------------
# CLI — quick sanity-check from the terminal
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a cheatsheet template by substituting equation placeholders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            '  python prompt.py "Does {{equation1}} imply {{equation2}}?" '
            '"x^2-1" "(x-1)(x+1)"'
        ),
    )
    parser.add_argument("prompt_text", help="Cheatsheet / prompt template text")
    parser.add_argument("equation1", help="First equation (replaces {{equation1}})")
    parser.add_argument("equation2", help="Second equation (replaces {{equation2}})")
    args = parser.parse_args()

    print(render_prompt(args.prompt_text, args.equation1, args.equation2))


if __name__ == "__main__":
    _main()
