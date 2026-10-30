"""Strands helpers shared by the use cases."""

from __future__ import annotations

import json
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager


def make_model(model_config: dict):
    """Policy model behind the rollout gateway, or an Amazon Bedrock model for comparison.

    model_config keys:
      base_url, api_key, model_id   -> OpenAI-compatible endpoint (rollout gateway)
      bedrock_model_id              -> Amazon Bedrock model
    """
    if model_config.get("bedrock_model_id"):
        from strands.models import BedrockModel

        return BedrockModel(model_id=model_config["bedrock_model_id"], temperature=model_config.get("temperature", 0.0))
    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        # max_retries=0: the gateway answers 429 when a rollout exceeds its turn cap.
        client_args={"api_key": model_config.get("api_key") or "EMPTY", "base_url": model_config["base_url"],
                     "max_retries": 0, "timeout": 600},
        model_id=model_config.get("model_id", "policy"),
        params=model_config.get("sampling_params", {}),
    )


def make_agent(model_config: dict, tools: list, system_prompt: str) -> Agent:
    """Return a Strands agent with the model from make_model(model_config), the given tools and system prompt.

    The agent keeps the full conversation (NullConversationManager) and prints nothing (callback_handler=None).
    """
    return Agent(
        model=make_model(model_config),
        tools=tools,
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )


def run(agent: Agent, prompt: str) -> tuple[str, str | None]:
    """Run the agent; return (final text, error). Errors end the episode instead of raising."""
    try:
        result = agent(prompt)
        content = result.message.get("content") or []
        return "".join(b["text"] for b in content if "text" in b), None
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"[:300]


def transcript(messages: list[dict], max_chars: int = 400) -> list[dict]:
    """Compact view of a Strands conversation: text, tool calls and tool results."""
    out = []
    for m in messages:
        for block in m.get("content") or []:
            if "text" in block and block["text"].strip():
                out.append({"role": m["role"], "text": block["text"][:max_chars]})
            elif "toolUse" in block:
                tu = block["toolUse"]
                out.append({"role": "call", "tool": tu.get("name"), "input": json.dumps(tu.get("input"))[:max_chars]})
            elif "toolResult" in block:
                tr = block["toolResult"]
                txt = " ".join(c.get("text", "") for c in tr.get("content", []) if isinstance(c, dict))
                out.append({"role": "result", "status": tr.get("status"), "text": txt[:max_chars]})
    return out


def count_turns(messages: list[dict]) -> int:
    """Return the number of assistant messages, which is the number of model calls."""
    return sum(1 for m in messages if m["role"] == "assistant")


def count_tool_calls(messages: list[dict]) -> int:
    """Return the number of tool calls in a Strands conversation."""
    return sum(1 for m in messages for b in m.get("content") or [] if "toolUse" in b)


def print_transcript(t: list[dict[str, Any]]) -> None:
    """Print a transcript from transcript(): tool calls as ->, tool results as <-, text with its role."""
    for e in t:
        if e["role"] == "call":
            print(f"  -> {e['tool']}({e['input']})")
        elif e["role"] == "result":
            print(f"  <- [{e.get('status')}] {e['text']}")
        else:
            print(f"  {e['role']}: {e['text']}")
