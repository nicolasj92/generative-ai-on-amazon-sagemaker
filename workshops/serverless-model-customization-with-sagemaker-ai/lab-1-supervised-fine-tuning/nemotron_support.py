"""
Nemotron support for the ContractNLI SFT lab.

Nemotron 3 Nano is a reasoning model. With the stock chat template its assistant
turn opens with `<think>\\n` and nothing closes it, so on this task the model spends
its whole output budget reasoning and never reaches the JSON answer. The managed
evaluation pipeline scores it 0 on every metric with 0% valid JSON.

Qwen3 does not have this problem because it obeys `/no_think`, which
`contractnli.build_prompt()` appends to the **user** turn. Nemotron ignores the
directive wherever it is put: system field, user turn, or both.

This module takes the dataset route. Every training completion begins with an empty
reasoning block, `<think>\\n</think>\\n`, so the model learns to close the block the
template opened and then answer. At inference the template opens `<think>\\n`, the
model emits `</think>` immediately, and the JSON follows. Nothing outside the
dataset changes: no template edits, no custom container, no inference-time flags.

Measured on the 123-contract held-out split:

    accuracy 84.6   evidence-F1 76.4   contradiction 77.7
    valid JSON 123/123   median response 938 chars

Two dataset variants were tried and rejected:

*   `system = "/no_think"` with an unmodified completion: 0.8% valid JSON. The
    directive alone does not change the model's behaviour.
*   A completion starting with only the closing tag, `</think>\\n\\n`: 0% valid JSON.
    Note this is what the trained model ends up emitting anyway, since the template
    supplies the opening tag, so why the target needs the pair is not established.

There is no `system` field here. An earlier version of this dataset carried
`system = "/no_think"` alongside the block and scored 84.2 against 84.6, a gap well
inside run-to-run variation.

Not the only fix
----------------

The same result is reachable without touching the dataset, by editing one line of
`chat_template.jinja` in the checkpoint:

    -{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
    +{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}

Applied to a checkpoint trained on plain JSON completions, that scores 84.5 / 75.1 /
76.6, indistinguishable from the numbers above. It renders `<think></think>` with no
newlines, so the two routes do not produce identical text. `enable_thinking` is not
exposed anywhere in the SDK, so the edit has to be made in S3 and the checkpoint
re-registered, and repeated for every new checkpoint. The dataset route travels with
the weights instead.

The base model
--------------

The base model has nothing trained in, so it reasons past the output cap and cannot
be scored as it stands. Its responses in the failed run ran 7,055 to 10,457
characters, every one cut mid-clause and none containing `{`, which puts them just
over a 2,048-token cap rather than far beyond it. Two routes exist, neither used
here: raise `max_new_tokens` through the evaluator's `overrides`, or copy the base
checkpoint out of JumpStart and patch its template.

For a base baseline the lab uses Bedrock (`nvidia.nemotron-nano-3-30b`), which
serves the model with reasoning off and returns JSON directly. That is a property of
Bedrock's serving stack, not obedience to the directive: the output is the same
whether `/no_think` is present or absent. Reasoning can be switched back on with
`additionalModelRequestFields={"chat_template_kwargs": {"enable_thinking": True}}`,
which takes the response from roughly 450 to 1,900 output tokens.
"""

from __future__ import annotations

# The empty reasoning block that every training completion starts with. The model
# learns to answer with the block already closed.
EMPTY_REASONING = "<think>\n</think>\n"


def training_record(prompt: str, answer: str) -> dict:
    """One SFT record for Nemotron, in the prompt/completion format."""
    return {
        "prompt": prompt,
        "completion": EMPTY_REASONING + answer,
    }


def eval_record(prompt: str, answer: str) -> dict:
    """One evaluation record. The reasoning block belongs to the model's turn, so the
    reference answer stays plain JSON."""
    return {
        "query": prompt,
        "response": answer,
    }
