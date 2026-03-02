# Socialite

Pub-sub multi-agent social learning bot with A/B strategy bidding.

Socialite is a 9-agent system that autonomously participates in the [Moltbook](https://www.moltbook.com) AI community. It observes community discussions, generates comments and posts, upvotes content, follows users, and continuously evolves its strategy through pattern mining and A/B testing.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start infrastructure
docker-compose up -d

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run one cycle
python runner.py --once

# 5. Run continuously (30-min interval)
python runner.py --interval 30
```

## Requirements

- Python >= 3.10
- [Ollama](https://ollama.ai) with `bge-m3` model (for embeddings)
- Docker (for Qdrant + Neo4j)
- OpenAI API key (primary LLM)
- Anthropic API key (fallback LLM, optional but recommended)
- Moltbook API key

## Architecture

Each cycle runs a 7-phase proposal-based pipeline:

```
Collect → Analyse → Propose → Coordinate → Execute → Learn → Report
```

### Subscription Topology

```
SensorAgent ──→ AnalysisAgent ──→ CommentAgent
                                ──→ PostAgent
                                ──→ UpvoteAgent
                                ──→ FollowAgent

Action Agents ──→ LearnerAgent
              ──→ ObserverAgent

LearnerAgent ──→ SensorAgent
             ──→ Action Agents
```

### 9 Agents

| Agent | LLM | Role |
|-------|-----|------|
| **SensorAgent** | - | Feed collection + API I/O gateway |
| **AnalysisAgent** | Haiku | Novelty scoring, quality assessment, topic extraction |
| **CommentAgent** | Haiku | Comment proposal generation with A/B strategy |
| **PostAgent** | Haiku | Post proposal generation with A/B strategy |
| **UpvoteAgent** | - | Upvote proposals based on engagement signals |
| **FollowAgent** | - | Follow/unfollow proposals based on community graph |
| **CoordinatorAgent** | - | Proposal arbitration (priority + budget + rate limits) |
| **LearnerAgent** | Haiku | Pattern mining + strategy evolution |
| **ObserverAgent** | - | A/B audit + cycle reporting |

### Core Framework (`core/`)

Custom async agent framework — no external agent library dependency.

| Module | Description |
|--------|-------------|
| `message.py` | `Message` dataclass (name, role, content, metadata, id, timestamp) |
| `base_agent.py` | `BaseAgent` with async `reply()`/`observe()` + subscriber fan-out |
| `msghub.py` | `MsgHub` pub-sub hub with selective subscription and broadcast |
| `llm.py` | `LLMClient` async wrapper (OpenAI primary, Anthropic fallback) |
| `proposal.py` | `Proposal` dataclass for action bidding |
| `ab_strategy.py` | `ABSelector` with alternate/probability modes |
| `message_logger.py` | Zero-intrusion agent message persistence to SQLite |

### Memory System (`social_memory/`)

Triple-backend memory architecture:

- **Vector Store** (Qdrant) — Semantic similarity search, novelty scoring
- **Structured Store** (SQLite/PostgreSQL) — Posts, comments, patterns, evolution state, agent messages
- **Knowledge Graph** (Neo4j) — Community relationships, PageRank, community detection

### Identity System (`SOUL.md`)

Three-layer identity constitution:

1. **Core** — Unchanging values and self-knowledge (never overwritten)
2. **Default Positions** — Stances that evolve with evidence
3. **Style Adaptation** — Dynamic context-based tone switching

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOLTBOOK_API_KEY` | Moltbook agent API key | (required) |
| `MOLTBOOK_AGENT_NAME` | Agent display name | `SocialLearnerBot` |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `ANTHROPIC_API_KEY` | Claude API key (fallback) | (optional) |
| `DAILY_BUDGET_USD` | Daily API spending limit | `5.00` |
| `QDRANT_HOST` | Qdrant vector DB host | `localhost` |
| `NEO4J_PASSWORD` | Neo4j graph DB password | (required) |
| `LOG_LEVEL` | Logging level | `INFO` |

See `.env.example` for the full list.

### A/B Strategy

Configured in `config.py` under `ab_strategy`:

- `alternate` mode — Strategies A and B alternate each cycle
- `probability` mode — Random selection with configurable `p_a`

### Evolution Stages

The system progresses through 4 stages: **Initial** → **Exploration** → **Optimization** → **Innovation**

- Cosine annealing in Exploration (exploration rate cycles 0.2–0.8)
- ReduceLROnPlateau in Optimization (reduces rate when engagement stalls)
- Catastrophic forgetting prevention (rollback if metrics drop >20%)

## Database Schema

18 tables across 4 layers:

| Layer | Tables |
|-------|--------|
| Community Data | `social_users`, `social_posts`, `social_comments` |
| Own Content | `own_posts`, `own_comments`, `feedback_snapshots` |
| Learning | `patterns`, `style_patterns`, `identity_snapshots`, `social_interactions` |
| Operations | `evolution_state`, `api_costs`, `behavior_log`, `agent_proposals`, `observer_reports`, `collection_strategy`, `planner_decisions`, `agent_messages` |

## Testing

```bash
uv run pytest tests/ -v
```

## Project Structure

```
socialite/
├── runner.py              # Main entry point — SocialiteRunner
├── config.py              # System configuration + evolution stages
├── SOUL.md                # Agent identity constitution
├── core/                  # Agent framework
│   ├── base_agent.py
│   ├── message.py
│   ├── msghub.py
│   ├── llm.py
│   ├── proposal.py
│   ├── ab_strategy.py
│   └── message_logger.py
├── agents/                # 9 specialized agents
│   ├── sensor_agent.py
│   ├── analysis_agent.py
│   ├── comment_agent.py
│   ├── post_agent.py
│   ├── upvote_agent.py
│   ├── follow_agent.py
│   ├── coordinator.py
│   ├── learner_agent.py
│   └── observer_agent.py
├── social_memory/         # Triple-backend memory
│   ├── storage_manager.py
│   ├── structured_store.py
│   ├── vector_store.py
│   ├── knowledge_graph.py
│   └── embeddings.py
├── moltbook/              # Moltbook API client
│   ├── client.py
│   ├── auth.py
│   ├── config.py
│   └── verifier.py
├── tests/
│   └── test_core.py
├── docker-compose.yml
└── .env.example
```

## License

Private project. All rights reserved.
