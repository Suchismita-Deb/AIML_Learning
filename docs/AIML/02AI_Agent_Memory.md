# AI Agent Memory

**Goal** - AI Agent needs memory - reliable behavior across long tasks and repeated sessions without blowing up context cost.

### Human Memory Analogy

- Human brain has mainly 4 types of memory - **_Short Term Memory, Factual Knowledge, Learned Skills, Personal Experience_**.

## CoALA Framework.

- The framework that AI used to get the memory is called **CoALA** - **Cognitive Architecture for Language Agents** points the 4 types of memory - **_Working memory, Semantic memory, Procedural memory and Episodic memory_**.

### Working memory.

- The context window of the LLM. Equivalent to RAM - fast, accessible and volatile.  
- Its limited in size and its removed when the session is over. The max today is 1M token in a context window. 
- Main limits - context window size, attention dilution, cost and latency.
  Core practice - keep only task-relevant context loaded.
- It starts at the beginning of every session.

### Semantic Memory

- It is the agents knowledge base and it stores facts and rules and conventions like the vector db. It mentions what the agent should know in general.
- It is simple a md file like the Claude.md
- Typical stores - docs, KB, wiki, vector index, structured metadata.
- Retrieval should be grounded with citations/source pointers where possible.  Risk - stale or conflicting facts if not versioned and refreshed.

### Procedural Memory

- It is the agents skill base and it tells the agent how to do things like the codebase. 
- It uses a file format called skill.md
- It mentions the skills and how to do the thing. Skills use progressive disclosure so the agent does not load all skills into the context window.  
- It loads the index and then when the task matches any skills then it loads the instructions. l
- Best when instructions are deterministic, scoped, and testable.

### Episodic Memory

- It is the agents personal experience and it tells the agent what it has done in the past.
- It uses a file format called experience.md
- Naive solution to store the transcript of all the conversation and use it. It is not a good solution.  
- It does some sort of distillations.
- The agent stores the distilled and compressed experience.
- Good episode schema - context, action taken, outcome, error/root cause, reusable lesson

## Common failure modes

1. Memory poisoning: bad data enters long-term store and keeps being reused.
2. Stale memory: old facts override current truth.
3. Over-retrieval: too much context causes weaker reasoning.
4. Under-retrieval: misses critical prior decisions.
5. Self-reinforcement loops: agent keeps trusting its own wrong summaries.

## Quick architecture pattern

1. User input.
2. Retrieve from semantic + procedural + episodic memory.
3. Assemble minimal working context.
4. Plan/act with tools.
5. Evaluate outcome.
6. Distill and write back high-value episode.

| Tool | Category | Notes |
|---|---|---|
| **Pinecone, Weaviate, Qdrant** | Vector DB | Managed, scalable, hybrid search support |
| **pgvector** | Postgres extension | Good for small-medium scale, co-locate with app DB |
| **Redis + RediSearch** | KV + vector | Fast, good for session memory |
| **LangChain Memory** | Abstraction layer | ConversationBufferMemory, SummaryMemory, VectorStoreMemory |
| **Mem0, Zep** | Managed memory | Purpose-built for agent memory, higher-level APIs |


## Context Window Management (critical for long sessions)

As conversation grows, you must decide what stays in context:

1. **Pin**: Goal, constraints, schema (immutable)
2. **Recent buffer**: Last N turns (recency bias)
3. **Retrieved**: Top-k relevant from long-term memory
4. **Summarized**: Older turns compressed to bullets

**Senior signal:** Define explicit eviction policy, not ad-hoc truncation.

## Memory Update Strategies

| Strategy | When to use |
|---|---|
| **Append-only** | Every turn goes to memory | Chat logs, audit trails |
| **Summarization** | Compress old turns before storing | Token efficiency, loses detail |
| **Entity extraction** | Pull facts/entities, store structured | Queryable knowledge base |
| **Reflection** | Agent decides what's worth remembering | Intelligent pruning, extra cost |
| **Forgetting** | TTL, LRU, or explicit user delete | Privacy, relevance, cost control |

## Retrieval Strategy.

**Naive:** Embed query, return top-k from vector DB.

**Production-grade:**
- Hybrid search (vector + keyword)
- Metadata filtering (time, user, domain)
- Re-ranking with cross-encoder
- Fusion (combine multiple retrievers)
- Query rewriting (expand/clarify before retrieval)
- Temporal decay (recent memory weighs higher)

**Key tradeoff:** Retrieval latency vs. relevance. P95 latency target drives architecture.