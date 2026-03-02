# CLAUDE.md — Socialite v0.4.0

## Commands

```bash
# Install dependencies
uv sync

# Install with all optional extras
uv sync --extra all

# Start infrastructure (Qdrant + Neo4j + optional PostgreSQL)
docker-compose up -d

# Run one cycle and exit
python runner.py --once

# Run continuously (default 30-min interval)
python runner.py --interval 30

# Run tests
uv run pytest tests/ -v
```

## Environment Setup

Copy `.env.example` to `.env` and set:
- `MOLTBOOK_API_KEY` — Moltbook agent API key
- `MOLTBOOK_AGENT_NAME` — Agent display name
- `OPENAI_API_KEY` — Primary LLM provider
- `ANTHROPIC_API_KEY` — Fallback LLM provider
- `DAILY_BUDGET_USD` — Daily API spending limit (default: $5.00)
- Qdrant/Neo4j/SQLite vars (see `.env.example`)

Embeddings use BGE-M3 via Ollama (localhost:11434). Vector dimension is 1024.

## Architecture

Socialite is a pub-sub multi-agent system with custom BaseAgent + MsgHub framework (no AgentScope dependency). Each cycle runs a proposal-based pipeline:

1. **Collect** — `SensorAgent` fetches Moltbook feed (multi-strategy)
2. **Analyse** — `AnalysisAgent` computes novelty/quality/topics → broadcasts to all subscribers
3. **Propose** — Action agents (`CommentAgent`, `PostAgent`, `UpvoteAgent`, `FollowAgent`) submit `Proposal` objects
4. **Coordinate** — `CoordinatorAgent` arbitrates proposals (priority + budget + rate limits)
5. **Execute** — Approved proposals executed via `SensorAgent` gateway
6. **Learn** — `LearnerAgent` mines patterns + updates strategy (periodic)
7. **Report** — `ObserverAgent` generates A/B reports (periodic)

### Subscription Topology

```
SensorAgent → AnalysisAgent → [CommentAgent, PostAgent, UpvoteAgent, FollowAgent]
Action Agents → [LearnerAgent, ObserverAgent]
LearnerAgent → [SensorAgent, Action Agents]
```

### 9 Agents

| Agent | File | LLM | Role |
|-------|------|-----|------|
| SensorAgent | `agents/sensor_agent.py` | None | Feed collection + API I/O |
| AnalysisAgent | `agents/analysis_agent.py` | Haiku | Data enrichment publisher |
| CommentAgent | `agents/comment_agent.py` | Haiku | Comment proposals with A/B |
| PostAgent | `agents/post_agent.py` | Haiku | Post proposals with A/B |
| UpvoteAgent | `agents/upvote_agent.py` | None | Upvote proposals |
| FollowAgent | `agents/follow_agent.py` | None | Follow/unfollow proposals |
| CoordinatorAgent | `agents/coordinator.py` | None | Proposal arbitration |
| LearnerAgent | `agents/learner_agent.py` | Haiku | Pattern mining + evolution |
| ObserverAgent | `agents/observer_agent.py` | None | A/B audit + reporting |

### Core Framework (`core/`)

- `message.py` — `Message` dataclass (name, role, content, metadata, id, timestamp)
- `base_agent.py` — `BaseAgent` with async reply/observe + subscriber fan-out
- `msghub.py` — `MsgHub` pub-sub with selective subscription and full-connect modes
- `llm.py` — `LLMClient` async wrapper (OpenAI primary, Anthropic fallback)
- `proposal.py` — `Proposal` dataclass for bidding (action, priority, strategy, target)
- `ab_strategy.py` — `ABSelector` with alternate/probability modes

### Copied Modules (from social-learner-bot v0.2.0)

- `moltbook/` — Moltbook API client (unchanged)
- `social_memory/` — Triple-backend memory (Vector + Structured + Graph)
- `SOUL.md` — 3-layer identity constitution

### Key Conventions

- All agents inherit from `core.BaseAgent`, not AgentScope
- Agent communication via `MsgHub` pub-sub, not direct calls
- A/B strategy built into each Action Agent via `ABSelector`
- `Proposal` objects replace direct action execution
- Line length: 100 characters
- `logging` module only, no `print()`
