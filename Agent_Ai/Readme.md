# Day 14 — AI Agents, ReAct Pattern & a Real-World Use Case

**Date:** 01 May 2025 (notes) — updated with current LangChain docs, Aug 2026

---

## 1. What is an AI Agent?

> An **AI agent** is an intelligent system that receives a **high-level goal** from a user, and **autonomously plans, decides, and executes a sequence of actions** by using external tools, APIs, or knowledge sources — all while **maintaining context, reasoning over multiple steps, adapting to new information, and optimizing for the intended outcome**.

```
AGENT  =  LLM  +  Memory  +  Tools
```

- **LLM** → the reasoning/decision-making core
- **Memory** → keeps context across steps (what's already been done/found)
- **Tools** → APIs, calculators, search, databases — the "hands"

### Core properties of an Agent

| Property | Meaning |
|---|---|
| **Goal-driven** | You tell the agent *what* you want, not *how* to do it |
| **Autonomous planning** | The agent breaks down the problem and sequences tasks **on its own** |
| **Tool-using** | Calls APIs, calculators, search tools, etc. as needed |
| **Context-aware** | Maintains memory across steps to inform future actions |
| **Adaptive** | Rethinks its plan when things change (e.g. an API fails, no data found, a place is closed) |

This is the difference between a normal LLM call and an agent: a plain LLM call answers once from a single prompt; an **agent loops**, re-planning as it goes, until the goal is actually achieved.

---

## 2. ReAct — Reasoning + Acting

> **ReAct** is a design pattern used in AI agents that stands for **Re**asoning + **Act**ing. It allows a language model (LLM) to **interleave internal reasoning (Thought)** with **external actions (like tool use)** in a structured, multi-step process.

Instead of generating an answer in one go, the model **thinks step by step**, deciding what it needs to do next, and **optionally calls tools** (APIs, calculators, web search, etc.) to help it.

### The ReAct loop

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
```

- **Thought** — the model's internal reasoning about what to do next
- **Action** — the tool it decides to call, and with what input
- **Observation** — the real result that comes back from that tool
- The loop **repeats** until the model has enough information to give a **Final Answer**

### Worked example — "What is the capital of France, and its population?"

```
Thought: I need to find the capital of France.
Action: search_tool
Action Input: "capital of France"
Observation: Paris

Thought: Now I need the population of Paris.
Action: search_tool
Action Input: "population of Paris"
Observation: 2.1 million

Thought: I now know the final answer.
Final Answer: Paris is the capital of France and has a population of ~2.1 million.
```

Simple query → LLM view:
```
query → [ LLM ] → response
```
Agent (ReAct) view — the LLM sits inside a **loop**, not a single call:
```
query → [ Thought → Action → Observation ] (repeats) → Final Answer
```

### ReAct is useful for
- **Multi-step problems**
- **Tool-augmented tasks** (web search, database lookup, etc.)
- **Making the agent's reasoning transparent and auditable** — you can literally read *why* it did what it did

### Origin
First introduced in the paper: **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022) — Princeton University & Google Research.

---

## 3. Agent & Agent Executor

Two separate pieces work together to run the ReAct loop:

| Component | Role |
|---|---|
| **Agent** | The "brain" — given the user query + history so far, decides the **next single step**: either an `Action` (call a tool) or `AgentFinish` (give the final answer) |
| **Agent Executor** | The "runtime" — actually **drives the loop**, calling the Agent repeatedly, running whatever tool it asked for, and feeding results back in |

> **AgentExecutor orchestrates the entire loop:**
> 1. Sends inputs and previous messages (the growing history/scratchpad) to the agent
> 2. Gets the next `action` from the agent
> 3. Executes that tool with the provided input
> 4. Adds the tool's `observation` back into the history
> 5. Loops again with the updated history **until the agent says `Final Answer`**

### Visualized flow

```
Agent Executor
      │
      ▼
Receive User Query
      │
      ▼
Pass User Query + Agent Scratchpad (Thought trail) to Agent
      │
      ▼
Agent Response? ──────────────┬───────────────────────
      │                                             │
      ▼                                             ▼
Agent Action                                  Agent Finish
(tool="search_tool", tool_input="...",     (return_values={"output": "..."},
 log="Thought: ...")                        log="Thought: I now know the
      │                                      final answer...")
      ▼                                             │
Execute Tool                                        ▼
      │                                       Return Final Output
      ▼
Collect Observation
      │
      ▼
Update Scratchpad ──── loop back to "Pass User Query + Scratchpad" ────┘
```

So the **Agent** only ever decides *one step at a time* — it's the **Agent Executor** that keeps calling it in a loop, running tools, and feeding results back, until it finally gets an `AgentFinish`.

⚠️ **Important update:** `AgentExecutor` is the **classic/legacy** LangChain pattern. In current LangChain (v1.0+), the recommended way to build an agent is `create_agent` (from `langchain.agents`), which runs this exact same *Thought → Action → Observation* loop internally, but implemented on top of **LangGraph** as an explicit graph/state machine instead of the old executor class. Conceptually everything above still applies — it's the same ReAct loop — the underlying implementation has just been modernized.

---

## 4. Creating an Agent

### What your notes show (older pattern)

```python
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)
```

```
LLM + Prompt → built from → downloaded from LangChain Hub (GitHub, prompt repository)

User Query + Thought trail → Agent → Action → Tool
                                   → Final Output → output
```

This came from `langgraph.prebuilt.create_react_agent` — a prebuilt, ready-to-use ReAct agent constructor that wires together an LLM, a list of tools, and a system prompt.

### ⚠️ Correction / current status

`langgraph.prebuilt.create_react_agent` is now **deprecated** in favor of **`create_agent`** from the `langchain` package, which offers the same agent loop plus a more flexible **middleware system** (for things like guardrails, dynamic tool selection, human-in-the-loop approval, etc.).

**Modern equivalent:**

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[search_tool],
    prompt=prompt,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the capital of France and its population?"}]
})
```

Internally, `create_agent` still runs the same loop your notes describe:

```
Setup: model + tools + prompt
   → Agent reasons (Thought)
   → Agent decides: call a tool (Action) or finish (Final Answer)
   → If tool called: Execute Tool → Observation → back to Agent
   → Repeat until Final Answer
```

So: **concepts (ReAct, Agent, Agent Executor/loop) are unchanged** — only the specific function names/imports have moved from `langgraph.prebuilt.create_react_agent` → `langchain.agents.create_agent`.

---

## 5. Real-World Use Case: AI Agent for Trip Planning

### The problem (traditional platforms)

Imagine a **60+ year old person** trying to plan a **budget-friendly 6–7 day trip to Goa** using today's typical booking sites. They run into several *separate, manual* problems:

1. **Travel/transport** — Car, bus, train, or flight? Which is cheapest? How does the choice affect the overall budget?
2. **Stay** — Which hotel? This usually means manually opening and comparing dozens of listings one by one.
3. **Itinerary** — What are the actual must-visit tourist places in Goa, and in what order does it make sense to see them?
4. **Local transport** — Once there, cab vs. rented scooty vs. some other local option — each with different cost/convenience trade-offs.

Each of these is its own research task. Done manually, it's slow, tiring, and easy to get wrong — especially for someone less comfortable juggling 5–6 different apps/tabs at once.

### How an AI-agent-powered platform changes this

Imagine a platform — say **bookrelaxed.com** — with an AI agent built in. The user just says:

> "Help me plan a 6–7 day affordable trip to Goa."

The **agent** (not a fixed workflow) then autonomously breaks this single high-level goal into the sub-problems above and works through them **itself**, exactly the way the "Autonomous planning" property described in Section 1 works:

| Step | What the Agent does |
|---|---|
| **1. Transport** | Calls travel-search tools/APIs for car, bus, train, and flight options for the user's dates → compares cost vs. time → surfaces options with a clear cost breakdown, rather than the user checking 4 separate sites |
| **2. Stay** | Searches hotels against the remaining budget after transport is chosen → filters for things relevant to a 60+ traveler (e.g. ground floor / elevator access, ratings) instead of a flat list to scroll through |
| **3. Itinerary** | Pulls together the actual popular Goa attractions and sequences them sensibly by day (using something like the itinerary tools shown in Day 13/14) |
| **4. Local transport** | Compares cab vs. rented scooter vs. public transport for getting between those attractions, and recommends based on cost + the user's comfort/mobility |
| **5. Confirm with user** | At each meaningful decision point (esp. cost-affecting ones), it **asks for confirmation** rather than silently booking — this matches the *"Context-aware"* property: it remembers prior answers so it doesn't re-ask the same thing |
| **6. Wrap-up** | Once confirmed, it can **email the full itinerary/invoice**, and **add the trip to the user's calendar** automatically |

### Why this beats a traditional platform

A traditional booking site is a **static tool** — it answers one query type at a time (flights *or* hotels *or* activities), and the *human* has to be the "agent," manually stitching results from 4-5 different searches into one coherent, budget-aware plan.

An **AI-agent-powered platform** instead:
- Takes **one high-level goal** instead of 5 separate manual searches
- **Plans the sequence itself** (transport → budget left → hotel → activities → local transport), the same way ReAct plans Thought → Action → Observation steps
- **Stays within a budget constraint across all decisions**, not just per-category
- **Reduces cognitive load** — genuinely valuable for a less tech-savvy or older user who doesn't want to compare 10 tabs
- **Closes the loop** — confirmation, invoicing, and calendar scheduling, not just "here are some search results"

### Handling exceptions — the "Adaptive" property in action

Your question about a **holiday falling mid-trip** (tourist places closed) is a great example of exactly the *Adaptive* property from Section 1:

> **Adaptive:** Rethinks plan when things change (e.g., API fails, no data)

Concretely, the agent would need to:
1. **Check** attraction opening hours / holiday calendars as part of building the itinerary (an additional tool call, e.g. a "local holidays/closures" lookup) — this is itself something the agent decides to do on its own, without being told to check for holidays
2. **Detect the conflict** — Day 3's planned attraction is closed that day
3. **Re-plan just that day** — swap in an alternative nearby attraction that *is* open, or reorder days so the affected attraction moves to a day it's open, rather than failing or leaving a blank day
4. **Re-confirm with the user** if the change is significant (e.g. "Attraction X is closed on the 15th for a local holiday — I've swapped it with Attraction Y on that day, and moved X to Day 5 instead. Want me to keep this?")

This is the same **Thought → Action → Observation → re-Thought** loop from Section 2 — the "Observation" here is simply "this place is closed," which triggers a new "Thought" and a revised "Action" (a new plan for that day), instead of the whole system just crashing or requiring the user to manually redo their itinerary.

---

## 6. Quick Summary

- **AI Agent** = LLM (brain) + Memory (context) + Tools (hands), driven by a **goal**, not step-by-step instructions
- **ReAct** = the Thought → Action → Observation loop that lets an LLM reason and act iteratively instead of answering in one shot
- **Agent** decides *one step at a time*; the **Agent Executor** (or, in current LangChain, the LangGraph-based loop inside `create_agent`) is what actually **drives the loop** until `Final Answer`
- `langgraph.prebuilt.create_react_agent` → now **deprecated**; use `langchain.agents.create_agent`
- **Real-world payoff:** an agent turns a messy, multi-step, manually-researched task (like planning an affordable Goa trip) into a single natural-language goal — the agent plans, checks budget/constraints across every step, adapts to real-world surprises (like holiday closures), and closes the loop with confirmation, invoicing, and calendar scheduling

---

### References
- "ReAct: Synergizing Reasoning and Acting in Language Models" — Yao et al., 2022
- https://docs.langchain.com/oss/python/langchain/agents
- https://www.langchain.com/blog/langchain-langgraph-1dot0