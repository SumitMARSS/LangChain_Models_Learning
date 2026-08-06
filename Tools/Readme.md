# Day 12 — Tools in LangChain (for Agents)

**Date:** 23 April 2025 (notes) — updated with current LangChain docs, Aug 2026

```
Tools ---> Tool Calling ---> Agents
```

---

## 1. Why do we need Tools?

An LLM has:
- **Reasoning** → it can *think*
- **Language generation** → it can *speak*

But an LLM has **no hands and legs** — it cannot act on the world by itself. It can't check today's weather, hit an API, run code, or query a database on its own.

**Tools** are what give the LLM "hands and legs." A tool is simple, self-contained logic (a Python function, or a wrapped API) that:
- Is packaged in a schema the LLM can understand
- Sits ready to be called when the LLM's reasoning decides it's needed
- Executes the actual action and returns a result back to the LLM

> A tool is just a Python function (or API) packaged in a way the LLM can understand and call when needed.

**LLMs are great at:** Reasoning (think), Language generation (speak)
**LLMs can't do:**
- Access live data (weather, news)
- Do reliable math
- Call APIs
- Run code
- Interact with a database

---

## 2. How Tools fit into the Agent ecosystem

> An **AI agent** is an LLM-powered system that can autonomously think, decide, and take actions using external tools or APIs to achieve a goal.

```
┌───────────────────────────────────────────────┐
│                    Agent                       │
│  ┌───────────────────────┐   ┌──────────────┐  │
│  │ Reasoning & Decision   │   │   Action     │  │
│  │       Making           │──▶│              │  │
│  │        (LLM)            │   │   (Tools)    │  │
│  └───────────────────────┘   └──────────────┘  │
└───────────────────────────────────────────────┘
```

- **LLM** = the brain (decides *what* to do)
- **Tools** = the hands (do the *actual* work)
- Together: LLM reasons → picks a tool → tool executes → result goes back to LLM → LLM reasons again (this loop is the **agent loop**)

---

## 3. Types of Tools

```
Tools
 ├── Built-in Tools   (pre-built, provided by LangChain)
 └── Custom Tools     (you define yourself)
```

### 3.1 Built-in Tools

> A built-in tool is a tool that LangChain already provides for you — pre-built, production-ready, and requires minimal or no setup. You don't have to write the function logic yourself — you just import and use it.

| Tool | Purpose |
|---|---|
| `DuckDuckGoSearchRun` | Web search via DuckDuckGo |
| `WikipediaQueryRun` | Wikipedia summary |
| `PythonREPLTool` | Run raw Python code |
| `ShellTool` | Run shell commands |
| `RequestsGetTool` | Make HTTP GET requests |
| `GmailSendMessageTool` | Send emails via Gmail |
| `SlackSendMessageTool` | Post message to Slack |
| `SQLDatabaseQueryTool` | Run SQL queries |

Full up-to-date list: **https://docs.langchain.com/oss/python/integrations/tools**

⚠️ **Status check (as of this update):** There is **no dedicated "built-in weather tool"** shipped by LangChain itself — weather has always been something you wrap yourself via a third-party API (OpenWeatherMap, WeatherAPI, etc.), either as a custom tool or through a community-contributed integration. Most of the built-in tools above (DuckDuckGo, Wikipedia, Python REPL, Requests, Gmail, Slack, SQL, etc.) historically lived in the `langchain_community` package:

```python
# Old way — still importable today, but on its way out
from langchain_community.tools import DuckDuckGoSearchRun
tool = DuckDuckGoSearchRun()
```

`langchain-community` is being **sunset/deprecated** as LangChain's core has been slimmed down for the v1.0 line. Integrations are being moved into standalone partner packages (e.g. `langchain-openai`, `langchain-chroma`) or into `langchain_classic` for older chain/retriever-style utilities. So:
- Old `langchain_community.tools.*` imports **still work for now** in most environments but throw deprecation warnings and shouldn't be relied on for new projects.
- The **recommended, current practice** is to just write your own thin `@tool`-wrapped function around whatever API you need (see below) instead of depending on community-maintained tool classes.

### 3.2 Custom Tools

> A custom tool is a tool that you define yourself.

**Use custom tools when:**
- You want to call **your own APIs**
- You want to **encapsulate business logic**
- You want the LLM to interact with **your database, product, or app**

#### Ways to create Custom Tools

```
                Ways to create Tools
                        │
      ┌─────────────────┼─────────────────┐
      ▼                 ▼                 ▼
 @tool decorator   StructuredTool      BaseTool
                    & Pydantic          class
```

| Approach | When to use |
|---|---|
| `@tool` decorator | Simplest, most common — quick functions with type hints |
| `StructuredTool` + Pydantic | When you need an explicit, more complex input schema |
| `BaseTool` subclass | Full control — custom validation, async behavior, advanced use cases |

---

## 4. Old way vs Modern way — Example: Weather Tool

### ❌ Old way (relying on a pre-built/community tool)

```python
from langchain_community.tools import SomeWeatherTool

tool = SomeWeatherTool()
```

Problems:
- You depend on LangChain/community maintaining it
- If the underlying API changes, the tool may silently break
- You have little control over the request and response shape
- The package itself (`langchain_community`) is being deprecated

### ✅ Modern way (custom tool wrapping your own API call)

```python
from langchain.tools import tool
import requests

@tool
def get_weather(city: str) -> str:
    """Returns current weather for a city."""
    url = f"https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={city}"
    response = requests.get(url)
    return response.json()
```

This is the pattern most people follow today: instead of trusting a pre-built LangChain tool, you wrap the API yourself and get full control.

> Note: current LangChain docs show `from langchain.tools import tool` (as well as `langchain_core.tools`) as the import path — both work, `langchain_core.tools` is the lower-level home, `langchain.tools` re-exports it for convenience.

**Type hints are required** — they define the tool's input schema. The **docstring becomes the tool's description**, which is what the LLM reads to decide when to use the tool.

---

## 5. How the LLM actually "sees" a tool

The LLM **never sees your tool's code**. It only sees the tool's **schema**:
- Tool **name**
- Tool **description** (from the docstring)
- Tool **input schema** (from type hints / Pydantic model)

The LLM decides *which* tool to call and *what arguments* to pass, purely based on this schema — the actual Python code runs on your side (server-side), and only the **result** is sent back to the LLM as a message.

```
LLM ──(sees schema: name, description, args)──▶ decides to call tool
LLM ──(tool_call: name + args)────────────────▶ your app executes the function
your app ──(result)────────────────────────────▶ back to LLM as a ToolMessage
LLM ──(reasons over the result)───────────────▶ final answer / next tool call
```

Reference: **https://docs.langchain.com/oss/python/langchain/tools**

### A few things that have evolved in current LangChain (good to know)

- **`ToolRuntime`** is the new unified way for a tool to access conversation state, context (e.g. user id), long-term memory (store), and streaming — replacing older patterns like `InjectedState`, `InjectedStore`, `get_runtime()`.
- Tools can **return**: a plain string, a structured object/dict, multimodal content (text + image blocks), or a `Command` object to directly update agent state.
- `return_direct=True` lets a tool's output skip going back through the model and be returned straight to the user — useful for simple lookups.
- **Dynamic tool selection**: the set of tools exposed to the model can be filtered/changed at runtime based on auth state, user role, or feature flags (via middleware), rather than being fixed at agent creation.
- **Headless tools**: tool definitions (schema only, no server-side implementation) that get executed on the *client* (e.g. browser APIs like geolocation) instead of your backend.

---

## 6. Toolkits

> A toolkit is just a collection (bundle) of related tools that serve a common purpose — packaged together for convenience and reusability.

**Example — `GoogleDriveToolKit`** might bundle:

| Tool | Purpose |
|---|---|
| `GoogleDriveCreateFileTool` | Upload a file |
| `GoogleDriveSearchTool` | Search for a file by name/content |
| `GoogleDriveReadFileTool` | Read contents of a file |

Toolkits save you from having to import and wire up 5–10 related tools individually — you just import the toolkit once.

---

## 7. Quick Summary

- LLM = brain (think + speak), Tool = hands (act)
- **Agent** = LLM + Tools working in a reasoning → action loop
- **Built-in tools** = pre-made by LangChain/community — fast, but you depend on someone else maintaining them, and many now live in the deprecating `langchain_community` package
- **Custom tools** = you write the logic (usually via `@tool`), wrapping your own APIs/business logic — this is now the more common, more resilient approach
- The LLM only ever interacts with a tool's **schema** (name, description, args) — never the underlying code
- **Toolkits** = pre-bundled groups of related tools for convenience

---

### References
- https://docs.langchain.com/oss/python/langchain/tools
- https://docs.langchain.com/oss/python/integrations/tools