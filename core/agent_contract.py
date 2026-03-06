# -*- coding: utf-8 -*-
"""Agent message contract helpers.

This module keeps agent I/O messages structurally consistent without forcing
business-level coupling across agents.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from core.message import Message

CONTRACT_VERSION = "1.0"
DEFAULT_INPUT_TYPE = "unspecified_input"
DEFAULT_OUTPUT_TYPE = "agent_reply"


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _safe_topic_segment(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return ""
    out = []
    for ch in text:
        if ch.isalnum() or ch in {".", "_", "-"}:
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("._-")


def normalize_incoming_message(agent_name: str, msg: Message) -> Message:
    """Normalize inbound message for predictable agent-side handling."""
    data = msg.to_dict()
    meta = _as_dict(msg.metadata)
    if not meta.get("type"):
        meta["type"] = DEFAULT_INPUT_TYPE
    contract_meta = _as_dict(meta.get("contract"))
    contract_meta.update({
        "schema_version": CONTRACT_VERSION,
        "kind": "input",
        "receiver": agent_name,
    })
    meta["contract"] = contract_meta
    if not data.get("topic"):
        inferred = _safe_topic_segment(meta.get("topic", "")) or f"agent.input.{agent_name}"
        data["topic"] = inferred
    data["metadata"] = meta
    return Message.from_dict(data)


def normalize_outgoing_message(
    agent_name: str,
    incoming_msg: Message,
    outgoing_msg: Message,
) -> Message:
    """Normalize outbound message for observability and contract stability."""
    data = outgoing_msg.to_dict()
    meta = _as_dict(outgoing_msg.metadata)
    message_type = _safe_topic_segment(meta.get("type", ""))
    if not message_type:
        message_type = f"{agent_name}.{DEFAULT_OUTPUT_TYPE}"
        meta["type"] = message_type
    contract_meta = _as_dict(meta.get("contract"))
    contract_meta.update({
        "schema_version": CONTRACT_VERSION,
        "kind": "output",
        "producer": agent_name,
        "input_type": (incoming_msg.metadata or {}).get("type", DEFAULT_INPUT_TYPE),
    })
    meta["contract"] = contract_meta
    meta.setdefault("source_agent", agent_name)
    meta.setdefault("input_message_id", incoming_msg.id)

    if not data.get("topic"):
        data["topic"] = _safe_topic_segment(meta.get("topic", "")) or f"agent.{agent_name}.{message_type}"
    # Preserve causal chain by default: reply inherits incoming trace.
    data["trace_id"] = incoming_msg.trace_id or incoming_msg.id
    if not data.get("causation_id"):
        data["causation_id"] = incoming_msg.id
    if data.get("target") is None:
        data["target"] = incoming_msg.name

    data["metadata"] = meta
    return Message.from_dict(data)


def parse_contract_json(raw: Any) -> Dict[str, Any]:
    """Parse JSON object from raw model output.

    Supports direct dicts, strict JSON strings, fenced JSON, or mixed text with
    a first JSON object embedded.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError("contract output must be dict or JSON string")

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        obj_text = _extract_first_json_object(text)
        if not obj_text:
            raise ValueError("no JSON object found in contract output")
        parsed = json.loads(obj_text)

    if not isinstance(parsed, dict):
        raise ValueError("contract output JSON must be an object")
    return parsed


def validate_required_fields(data: Dict[str, Any], required_fields: Iterable[str]) -> Tuple[bool, List[str]]:
    """Validate required top-level fields for prompt contracts."""
    required = [f for f in required_fields if f]
    missing = [field for field in required if field not in data]
    return len(missing) == 0, missing


def _extract_first_json_object(text: str) -> str:
    start = -1
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return ""
