# Harness Engineering

## The Evolution: Prompt → Context → Harness

```
Prompt Engineering  →  Context Engineering  →  Harness Engineering
    (2022–2023)            (2023–2024)               (2024–present)
```

Each layer was born because the previous one hit a hard ceiling at scale.

---

## 1. Prompt Engineering

When ChatGPT launched, the context window was **~4,096 tokens**. The entire discipline was about compression — how do you fit everything the model needs into ~3,000 words?

Key techniques: few-shot examples, chain-of-thought, role prompting, instruction tuning.

**The ceiling it hit:** Every call is stateless. No tools, no memory, no external knowledge. Works for isolated Q&A but breaks completely for multi-step, real-world tasks.

---

## 2. Context Engineering

As windows grew (8K → 32K → 128K → 1M tokens), the problem shifted from *fitting* data to *managing what goes in* dynamically.

**Context Engineering** dynamically constructs the context window at runtime using three mechanisms:

| Mechanism | What it does |
|---|---|
| **Tool Calling** | LLM invokes functions (search, read file, run code) to pull only what it needs |
| **RAG** | Vector database retrieval injects relevant documents on demand |
| **MCP (Model Context Protocol)** | Vendor-level protocol that plugs external services directly into the model |

**Tools in practice:** Cursor, Windsurf, Cline, Aider — all IDE agents built on tool calling. LangChain and LlamaIndex for RAG pipelines.

### The Context Collapse Problem

This is the failure mode Context Engineering cannot fix on its own.

When a complex, long-running task fills the context window, the model is forced to **summarize its own history** to reclaim space. This causes:

- **Over-summarization** — fine-grained details are compressed away
- **Illusion of completion** — the model believes it finished subtasks it actually truncated
- **Goal drift** — later steps lose fidelity to the original intent

At FAANG scale, this is the equivalent of a distributed worker that silently corrupts its own job queue under memory pressure — catastrophic and hard to detect.

---

## 3. Harness Engineering

**Harness Engineering** is the design of the *environment, orchestration rules, and state management* that allow agents to complete long-horizon tasks reliably — beyond what any single context window can hold.

You are not engineering the prompt. You are engineering the **system the agent runs inside**.

---

## Core Concepts

### Agent Loops

An agent loop is a repeating execution cycle:

```
Task → [Think → Act → Observe] × N → Checkpoint / Exit
```

Each iteration: the agent reasons, calls a tool or produces output, observes the result, updates state, then decides whether to continue or terminate.

**Critical property:** The loop must be **restartable from any checkpoint**. This is the same guarantee you require from any distributed worker — idempotency and at-least-once delivery semantics. An agent loop that cannot resume is a prototype, not a production system.

---

### Hierarchical Context Management

Instead of one agent accumulating an ever-growing context, tasks are decomposed into a **tree of sub-tasks**, each executed by a sub-agent with a fresh, scoped context window.

```
Orchestrator  (high-level plan, minimal context)
├── Sub-Agent A  →  "Research topic X"      [fresh 8K context]
├── Sub-Agent B  →  "Write section Y"       [fresh 8K context]
└── Sub-Agent C  →  "Validate output Z"     [fresh 8K context]
```

Each sub-agent only sees what it needs. Failures are isolated. Sub-agents can run in parallel.

**The systems analogy is exact:** This is MapReduce. Decompose → distribute → aggregate. The orchestrator is the job scheduler, sub-agents are workers, and the context window is each worker's RAM budget.

---

### Sub-Agent Orchestration Patterns

| Pattern | How it works | Use when |
|---|---|---|
| **Sequential** | Task B starts after Task A completes | B depends on A's output |
| **Parallel** | Tasks run simultaneously | Tasks are independent |
| **Conditional** | Next task chosen based on prior result | Branching workflows |
| **Hierarchical** | Sub-agents spawn their own sub-agents | Deep, recursive tasks |

**FAANG example:** A production coding agent orchestrates a planner → code writer → test writer → reviewer → integrator — each as a separate agent call with scoped context, running sequentially where dependent and in parallel where independent.

---

### Swarm Architecture

A **swarm** runs agents with minimal central coordination. Each agent follows local rules and self-selects tasks, producing emergent collective behavior without a single orchestrator.

| | Hierarchical | Swarm |
|---|---|---|
| **Control** | Centralized orchestrator | Decentralized, peer-to-peer |
| **Failure mode** | Orchestrator is a single point of failure | Degrades gracefully |
| **Best for** | Well-defined, structured workflows | Exploratory, open-ended tasks |

Frameworks: AutoGen (Microsoft), CrewAI, OpenAI Swarm.

---

### Checkpointing and Resumption

Long-running agent tasks — measured in minutes or hours — will encounter API failures, rate limits, context overflow, and states requiring human review. Without checkpointing, a failure at step 47 of 50 means starting over.

**Checkpointing** persists the agent's state — completed steps, intermediate outputs, current task position — so execution resumes from the last known good state.

```
Step 1 ✅ [checkpoint]  →  Step 2 ✅ [checkpoint]  →  Step 3 ❌
                                                          ↓
                                              Resume from Step 2
                                                          ↓
                                              Step 3 (retry) ✅
```

**Implementation patterns:**
- Persist checkpoints to a durable store (Redis, DynamoDB, S3)
- Use **event sourcing** — every action is an immutable append-only log entry
- All steps must be **idempotent** — re-executing a completed step produces the same result, no side effects

This is write-ahead logging for agents. Any senior engineer who has designed fault-tolerant distributed systems will recognize it immediately — and that recognition is exactly the signal FAANG interviewers look for.

---

## Harness vs Context Engineering

| Dimension | Context Engineering | Harness Engineering |
|---|---|---|
| **Unit of work** | Single LLM call | Multi-step agent loop |
| **Memory** | One context window | Distributed across agents |
| **Failure handling** | Prompt retry | Checkpoint + resume |
| **Scale** | One model, one task | Many agents, parallel execution |
| **Key engineering skill** | Retrieval + prompt design | Systems design + orchestration |

---

## Failure Modes to Know

| Failure Mode | What happens | Mitigation |
|---|---|---|
| **Context drift** | Agent loses alignment with original goal over many iterations | Re-inject the goal statement at each checkpoint |
| **Runaway loops** | Agent loops infinitely on a stuck tool call | Hard iteration cap + loop detection heuristic |
| **Over-summarization** | Compression destroys critical detail | Structured summaries with pinned immutable key facts |
| **Sub-agent hallucination** | Sub-agent returns confident, wrong output | Validator agent + confidence threshold gating |
| **Cascading failure** | One sub-agent failure corrupts downstream agents | Isolation boundaries + explicit fallback paths |
| **Prompt injection** | Malicious content in retrieved data hijacks agent behavior | Input sanitization + sandboxed tool execution |

---

## Key Tools & Frameworks

| Tool | Category | What it gives you |
|---|---|---|
| **LangGraph** | Orchestration | Stateful, graph-based agent workflows with built-in checkpointing |
| **AutoGen** | Multi-agent | Conversational multi-agent coordination |
| **CrewAI** | Role-based agents | Agents with defined roles, goals, and tool access |
| **Temporal** | Durable workflows | Production-grade fault-tolerant workflow orchestration (not AI-specific, highly applicable) |
| **OpenAI Assistants API** | Managed agents | Thread + run model with native state persistence |

---

## Mental Model for Interviews

> *"Harness Engineering applies the same principles distributed systems use to coordinate unreliable workers — decomposition, isolation, idempotency, checkpointing, and fault tolerance — to the problem of running LLM agents on complex, long-horizon tasks."*

Map every harness concept to a systems concept you already know:

| Agent concept | Systems equivalent |
|---|---|
| Agent loop | Worker process / consumer loop |
| Context window | Working memory / RAM budget |
| Checkpoint | Write-ahead log / commit point |
| Sub-agent | Microservice / worker shard |
| Orchestrator | Workflow engine (Airflow, Temporal) |
| Swarm | Peer-to-peer distributed system |
