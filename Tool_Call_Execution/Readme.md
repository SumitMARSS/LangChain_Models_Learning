# Day 13 — Tool Calling, Tool Execution & Injected Tool Args

**Date:** 25 April 2025 (notes) — updated with current LangChain docs, Aug 2026

```
Tool Creation → Tool Binding → Tool Calling → Tool Execution → (loop) → Agent
```

---

## 1. Tool Calling

> **Tool Calling** is the process where the **LLM** decides, during a conversation or task, that it needs to **use a specific tool (function)** — and generates a **structured output** with:
> - the **name** of the tool
> - the **arguments** to call it with

🚨 **Important:** The LLM does **not** actually run the tool — it only **suggests** the tool name + input arguments. The **actual execution** is handled by LangChain (or your own code).

### Example

**User:** "What's 8 multiplied by 7?"

The LLM responds with a **tool call** (not an answer):

```json
{
  "tool": "multiply",
  "args": { "a": 8, "b": 7 }
}
```

- `name` → `multiply`
- `args` → matches the tool's input **schema**

This is why, before Tool Calling can happen at all, two setup steps are needed:

```
[ Tool Creation ]   → writing the @tool function itself
[ Tool Binding  ]   → attaching / registering the tool(s) with the LLM
                       (e.g. llm.bind_tools([multiply, get_weather]))
```

Binding is what makes the LLM *aware* a tool exists and shows it that tool's schema — without binding, the LLM has no tools to call at all.

---

## 2. Tool Execution

> **Tool Execution** is the step where the **actual Python function (tool)** is run using the input arguments that the **LLM suggested** during tool calling.

In simpler words:

🧠 **The LLM says:**
"Hey, call the `multiply` tool with `a=8` and `b=7`."

⚙️ **Tool Execution** is when *you* (or LangChain) actually run:

```python
multiply(a=8, b=7)
```
→ and get the result: `56`

That result is then sent **back to the LLM** as a `ToolMessage`, so the LLM can use it to form its final, natural-language answer ("8 multiplied by 7 is 56.").

### Tool Calling vs Tool Execution — the key difference

| | Tool Calling | Tool Execution |
|---|---|---|
| **Who does it** | The LLM | LangChain / your code |
| **What happens** | LLM *suggests* a tool name + args | The suggested function actually *runs* |
| **Output** | A structured tool-call object (JSON-like) | The real result (e.g. `56`) |

The LLM only ever *proposes* an action. It's your application (or the agent loop) that carries it out.

---

## 3. Injected Tool Arguments

Normally, **every parameter** in a tool's function signature becomes part of the schema the LLM sees, and the LLM has to guess/fill in a value for it. But sometimes you **don't want the LLM to supply a value itself** — because:
- it's sensitive (API keys, user IDs, tokens),
- it's something only your code knows (a session, a DB connection),
- or — the scenario you raised — **it should only ever come from the result of a previous tool call**, never from the LLM's own "knowledge" or guess.

For this, LangChain provides `InjectedToolArg`:

> `InjectedToolArg` marks a tool argument as **injected at runtime**. Arguments annotated with it are **removed from the schema sent to the LLM** — the LLM never sees them, is never asked to fill them, and cannot supply a value for them. Instead, **your code injects the real value** right before the tool executes.

### Example — currency conversion

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Given a currency conversion rate, this function calculates the
    target currency value from a given base currency value."""
    return base_currency_value * conversion_rate
```

Here:
- `base_currency_value` → **visible to the LLM** (it's part of the tool schema; the LLM fills this in, e.g. `10`)
- `conversion_rate` → **hidden from the LLM** (marked `InjectedToolArg`) — the LLM is never even shown this parameter, so it **cannot hallucinate a rate**. Your application must supply it — typically with the real value returned by another tool.

---

## 4. Your scenario: forcing sequential tool execution

### The problem you identified

Suppose you have two tools:
1. `get_conversion_factor` → fetches the **live** USD→INR exchange rate
2. `convert` → multiplies a base amount by a rate

User asks:
> "What is the current USD to INR exchange rate, and can you convert 10 dollars to rupees?"

**Risk:** Since the LLM already "knows" (from its training data) roughly what USD-to-INR conversion looks like, there's a real chance the LLM tries to answer `convert` using a **guessed/remembered** rate from its own parametric knowledge — *before* `get_conversion_factor` has even actually run and returned a real, current number. That gives you a wrong, stale answer that looks confident.

### The fix: `InjectedToolArg`

By marking `conversion_rate` as `InjectedToolArg`:
- The LLM's tool schema for `convert` **no longer includes `conversion_rate` at all**
- The LLM literally **cannot** pass a value for it — there's nothing in the schema to fill
- Your orchestration code is now **forced** to plug in the value — and the only place you'd reasonably get that value from is the **actual result of `get_conversion_factor`**

This gives you, the developer, **control over the sequence**:

```
1. User says: "Convert 10 USD to INR."
2. LLM thinks: "I don't know the rate. First, let me call get_conversion_factor."
3. Tool result comes: 85.3415
4. LLM looks at result, THINKS again: "Now I know the rate, next I should call
   convert with 10 and 85.3415."
5. Tool result comes: 853.415 INR
6. LLM summarizes: "10 USD is 853.415 INR at the current rate."
```

Step 4 is the important one — because `conversion_rate` is injected, the value going into `convert` **must** be the real number that came out of `get_conversion_factor` in step 3 (your code passes it through), not a number the LLM invented. This is exactly how you enforce **"tool B only runs using tool A's real output"** rather than letting the model's own knowledge sneak into the answer.

> In short: `InjectedToolArg` isn't just about hiding secrets from the LLM — it's also a practical way to **enforce correctness and ordering** between tools that depend on each other's live results.

---

## 5. Does Tools + Tool Calling = Agent?

**No.**

Tools + Tool Calling only give an LLM the *ability* to request an action and get a result back. On their own, they're just a single request → single tool → single result loop, still driven by whatever code you wrote to wire it together.

> An **Agent** is what you get when the LLM is put in a loop where, **without you manually telling it what to do at each step**, it can:
> 1. **Break down** a bigger/ambiguous problem into smaller steps on its own
> 2. **Decide** which tool (if any) is needed for the current step
> 3. **Call** that tool (Tool Calling)
> 4. **Observe** the result (Tool Execution)
> 5. **Re-think** based on that result, and decide the next step
> 6. **Repeat** until the goal is achieved, then produce a final answer

This is the **agent loop**, and it's exactly what the currency-conversion example (Section 4) demonstrates: the LLM wasn't told "first call tool A, then call tool B" by a human — it figured out on its own, step by step, that it needed the exchange rate before it could convert. That autonomous, self-directed planning across multiple steps — not just the presence of tools — is what makes it an **agent**.

```
Tool Creation + Tool Binding + Tool Calling + Tool Execution
                        │
                        ▼
        (wrapped in an autonomous think → act → observe → repeat loop)
                        │
                        ▼
                     Agent
```

---

## 6. Quick Summary

- **Tool Calling** = LLM *suggests* a tool name + args (structured output, no execution)
- **Tool Execution** = your code / LangChain actually *runs* that function and gets a real result
- **Injected Tool Args** (`InjectedToolArg`) = parameters hidden from the LLM's schema; the LLM can never fill them — your code injects the real value at execution time
  - Useful for secrets/context (API keys, user IDs)
  - Also useful for **enforcing sequencing**: forcing a downstream tool to only run with the *real* output of an upstream tool, instead of the LLM guessing a value from its own training knowledge
- **Tools + Tool Calling alone ≠ Agent.** An **Agent** is the autonomous loop where the LLM breaks a problem into steps and drives itself through think → call tool → observe result → think again, without step-by-step human instruction.

---

### References
- https://docs.langchain.com/oss/python/langchain/tools
- https://reference.langchain.com/python/langchain-core/tools (see `InjectedToolArg`, `InjectedToolCallId`)