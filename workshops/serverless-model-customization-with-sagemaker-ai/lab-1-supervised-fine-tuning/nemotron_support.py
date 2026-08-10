"""
Nemotron support for the ContractNLI SFT lab.

Nemotron 3 Nano is a reasoning model: its chat template opens an unterminated
`<think>` block at the start of every assistant turn. On this task it then spends
its whole output budget reasoning and never reaches the JSON answer, so the
managed evaluation pipeline scores it 0 on every metric with 0% valid JSON.

Qwen3 does not have this problem because it *obeys* `/no_think` in the system
prompt and emits an empty reasoning block. Nemotron ignores the instruction, so
the behaviour has to be trained in.

The fix is two matching pieces:

1.  Every training prompt carries `system = "/no_think"`, declaring the intent
    through the documented SFT `system` field.
2.  Every training completion begins with an empty reasoning block,
    `<think>\n</think>\n`, which is byte-for-byte what the template renders when
    `enable_thinking=False`. The model learns to answer with the reasoning block
    already closed.

At inference the template opens `<think>\n`, the model closes it immediately, and
the answer follows. Nothing outside the dataset changes: no template edits, no
custom container, no inference-time flags.

Measured on the 123-contract held-out split (`nemotron-contractnli-v3-mpg`):

    accuracy 84.2   evidence-F1 76.0   valid JSON 123/123   median 938 chars

Two variants were tried and rejected:

*   `system = "/no_think"` with an unmodified completion — 0.8% valid JSON. The
    system field alone does not change the model's behaviour.
*   A completion starting with only the closing tag, `</think>\n\n` — 0% valid
    JSON. The model needs to see the complete `<think>...</think>` pair to learn
    the pattern.

The base model cannot be evaluated through the managed pipeline at all: with
nothing trained in, it reasons past the output cap whatever the system prompt
says. Use Bedrock (`nvidia.nemotron-nano-3-30b`) with
`system=[{"text": "/no_think"}]` for the base baseline, where the serving stack
honours the directive.
"""

from __future__ import annotations

import json

# Declares the intent through the documented SFT `system` field.
SYSTEM_PROMPT = "/no_think"

# Exactly what the chat template emits for enable_thinking=False, so the model
# learns to answer with the reasoning block already closed.
EMPTY_REASONING = "<think>\n</think>\n"


def training_record(prompt: str, answer: str) -> dict:
    """One SFT record for Nemotron, in the prompt/completion format."""
    return {
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "completion": EMPTY_REASONING + answer,
    }


def eval_record(prompt: str, answer: str) -> dict:
    """One evaluation record. The reasoning block belongs to the model's turn,
    so the reference answer stays plain JSON."""
    return {
        "system": SYSTEM_PROMPT,
        "query": prompt,
        "response": answer,
    }


def write_dataset(path, docs, labels, gold_for, build_prompt, split):
    """Write a JSONL dataset for one split. `split` is 'train'/'val' or 'test'."""
    keys = list(labels.keys())
    make = eval_record if split == "test" else training_record
    with open(path, "w") as handle:
        for doc in docs:
            gold = gold_for(doc)
            answer = json.dumps({k: {"label": gold[k]["choice"],
                                     "evidence": list(gold[k]["spans"])}
                                 for k in keys if k in gold})
            handle.write(json.dumps(make(build_prompt(doc, labels), answer)) + "\n")
