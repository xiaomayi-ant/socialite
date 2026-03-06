# -*- coding: utf-8 -*-
"""Socialite core framework — BaseAgent, MsgHub, LLM, Proposal, A/B."""

from core.message import Message
from core.base_agent import BaseAgent
from core.msghub import MsgHub
from core.llm import LLMClient
from core.proposal import Proposal
from core.ab_strategy import ABSelector
from core.message_logger import MessageLogger
from core.decision import AgentDecision
from core.agent_contract import (
    CONTRACT_VERSION,
    normalize_incoming_message,
    normalize_outgoing_message,
    parse_contract_json,
    validate_required_fields,
)
from core.rate_limit import RateLimitPolicy
from core.replay import (
    DecisionSnapshot,
    compare_decision_snapshots,
    load_decision_snapshots,
    compare_cycles,
    build_cycle_replay_report,
)

__all__ = [
    "Message", "BaseAgent", "MsgHub", "LLMClient",
    "Proposal", "ABSelector", "MessageLogger", "AgentDecision", "RateLimitPolicy",
    "CONTRACT_VERSION", "normalize_incoming_message", "normalize_outgoing_message",
    "parse_contract_json", "validate_required_fields",
    "DecisionSnapshot", "load_decision_snapshots", "compare_decision_snapshots", "compare_cycles",
    "build_cycle_replay_report",
]
