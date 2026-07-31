# LangChain_Models_Learning

## Overview

This repository documents my hands-on journey learning LangChain—a framework for building powerful language model applications. On the first day, I explored why LangChain is needed, the concept of semantic search, its main components, and the benefits of adopting this library.

## 📅 Day 1 - Introduction of Langchain

### Why LangChain?

LangChain enables rapid development of applications using LLMs (Large Language Models), simplifying tasks such as text understanding, generation, and workflow automation. It abstracts away low-level details, allowing developers to focus on higher-level logic and integration.

### 🌟 Key Benefits of LangChain

- **🧩 Concept of Chains:** Lets you link multiple LLM steps together (like prompt → model → parser) to build complex workflows easily.
- **🌐 Complete Ecosystem:** Provides built-in tools, integrations, and utilities (agents, vector stores, retrievers) to build end-to-end LLM apps quickly.
- **🧠 Memory & State Handling:** Supports conversational memory to remember previous chats and maintain context across multiple turns.
- **⚙️ Model-Agnostic Development (Component Modularity):** Works with any LLM (OpenAI, Claude, Gemini, HuggingFace, etc.) using the same interface—making it easy to swap models without changing your code.


### Components of LangChain

- **Models:** LLM providers (OpenAI, Hugging Face, etc.) for standard and custom text generation.
- **Prompts:** Templates and formatting for model inputs—configurable for different tasks and fine-tuning.
- **Chains:** Sequential pipelines linking models, prompts, and external tools to enable complex workflows.
- **Memory:** Mechanisms for passing context/history between chain steps and improving result relevance.
- **Indexes:** Data storage and retrieval systems designed for semantic search, chunking, and fast access.
- **Agents:** Autonomous decision-making units that dynamically select tools or chains based on task requirements.

## 📅 Day 2 - Prompt, Chatbots

### 1. Prompt Engineering

- **Static Prompts**: Hardcoded text prompts that remain the same every time.
- **Dynamic Prompts**: Templates with variables that are filled at runtime.
- **Prompt Structures Tried**:
  - **Single Message**: A single user/system message sent to the LLM.
  - **List of Messages**: A sequence of system, user, and assistant messages for multi-turn context.
  - **Mixed Static + Dynamic Messages**: Combining static instructions with runtime user input dynamically.

#### 📝 Note: :- I had used Streamlit for some UI interaction while learning prompt engineering so that I can visualize things in an easy way.
      To install it, run:  pip install streamlit

---

### 2. Interacting with LLMs

- Practiced integration with **closed-source LLMs** like:
  - OpenAI GPT
  - Claude (Anthropic)
  - Gemini (Google)
- Practiced integration with **open-source LLMs** via:
  - Hugging Face Hub APIs

This gave hands-on experience on how to build and run inference calls across multiple model providers.

---

### 3. Building Chatbots

- Built **basic chatbots** using LangChain's LLMChain.
- Created **messaging-based chatbots** using ChatPromptTemplate and ChatOpenAI.
- Implemented a **support chatbot with memory**, capable of remembering previous user messages across turns using ConversationBufferMemory.

**File:** `supportChatWithMemory.py`

---

### 4. Message Placeholders

- **Message placeholders** are variables inside a message template that get replaced with actual values at runtime.
- They allow injecting dynamic user input while keeping system instructions consistent.
- Used in `supportChatWithMemory.py` to:
  - Preserve previous conversation context (`chat_history`)
  - Accept new user input dynamically (`user_input`)

---

### 5. Semantic Search

- Integrated **embedding models** from Hugging Face to convert text into vector embeddings.
- Stored embeddings in a vector store (FAISS/Chroma) and used **similarity search** to retrieve the most relevant content for a query.
- This approach is called **semantic search**, as it retrieves based on meaning instead of exact keyword match.

**Example use case:**  
Given a user query like *"symptoms of flu"*, the system fetches documents related to flu even if the word “flu” is not explicitly present but semantically similar.


## 📅 Day 3 – Parsing Techniques

### 🧠 Why Parsing is Needed
When using LLMs, the model’s responses are plain text.  
To use this output in code (for further processing, storing in DB, visualizing, etc.), we need to **parse it into structured data formats** like JSON or Python objects.  
LangChain provides parsing utilities to help enforce structure on LLM outputs.

---

### 1. Structured Output Parsing

- Ensures the LLM always returns output in a **specific, well-defined structure**.
- Helps prevent runtime errors and makes the output easy to use in code.

**Common Techniques:**
- **TypedDict**: Defines a Python dictionary with expected keys and types.
- **Pydantic Models**: Defines a strict schema with field types and validation.
- **JSON Schema**: Defines structure using JSON Schema format.

**Example Use Case:**
- Ask LLM: “Extract name and age from this sentence.”  
- Get result as:
  ```json
  {
    "name": "John",
    "age": 25
  }

### 2. Unstructured Output Parsing

- Used when the LLM can return free-form text, and you want to post-process it into structured data.
- More flexible than structured parsing, but requires parsing logic after the LLM output.

**Main Parsers:**
- **StrOutputParser**: Returns raw text output from the LLM.
- **JsonOutputParser**: Parses JSON-like output from the LLM.
- **StructuredOutputParser**: Uses a predefined schema to parse text into structured objects.
- **PydanticOutputParser**: Parses output directly into Pydantic models.

**Example Use Case:**
- LLM returns: "John is 25 years old."
- Use JsonOutputParser or regex with StrOutputParser to extract:
  {"name": "John", "age": 25}

## 📅 Day 4 – Chains

### 🪢 What are Chains in LangChain?

**Chains** are just **multiple Runnables connected together** to form a **workflow**.

- Each step is a `Runnable`.
- You combine them with `|` or other composition utilities.
- The output of one step becomes the input of the next.

This makes it easy to build complete AI systems from small pieces.

---

### ⚙️ Types of Chains

#### 1. Sequential Chains
**Run steps one after another.**  
The **output of step A goes to step B**, then to step C, and so on.

**Use case:** Step-by-step workflows like:
- Create a prompt → Send to LLM → Parse answer


#### 2. Parallel Chains
**Run multiple steps at the same time on the same input.**
Then combine their outputs into one object.
**Use case:** When you want to get multiple different responses or extract different fields in one go.


#### 3. Conditional Chains
**Choose which chain to run based on input or logic.***
Use case: When you want to branch the flow depending on conditions.


### 🧠 What is Runnable in LangChain?
  In LangChain, **Runnable** is like a universal building block.  
  It’s a **common interface (a shared structure)** that **wraps any step in your AI pipeline** so they can all work together in the same way.

Think of Runnable as a **plug adapter**:

- You can plug in a **prompt template**
- Or an **LLM call**
- Or a **tool**
- Or a **custom Python function**

Once something is a Runnable, you can:

- **Run it** (`.invoke()`)
- **Run many at once** (`.batch()`)
- **Stream outputs as they come** (`.stream()`)

This gives everything the **same controls, same methods, and same behavior.**

---

### ⚡ Why Runnable is Needed

**Without Runnable:**

- Every step (LLM, prompt, tool, parser) has **different ways of being called**
- You need **extra code** to connect them together
- Pipelines become **messy and hard to debug**

**With Runnable:**

- You get **one simple interface** for every component
- You can **chain steps together easily** using the `|` operator (like building blocks)
- You can **add logic, retry, error handling, logging** in one place
- You can **switch or reuse components** without changing your whole pipeline

> 📝 In short: it removes confusion and boilerplate, making code cleaner.

---

### 🧩 How Runnable Solves Developer Problems

    | Developer Problem                  | How Runnable Helps                                    |
    |-----------------------------------|---------------------------------------------------------|
    | Hard to connect different steps    | Makes all steps follow same interface                   |
    | Messy async / batch code           | Built-in `.batch()` and `.stream()` support              |
    | Difficult to test or replace pieces| Any piece can be swapped easily if it’s a Runnable       |
    | Complex pipelines                  | Can combine with `|` like Lego blocks                    |
    | Hard to debug                       | Central place to add tracing, logging, retries           |

---


## 📅 Day 5 –Understanding Runnables in LangChain

## 📌 Background

Earlier, the LangChain team built **separate components** for each step in an LLM pipeline, like:

- Output parsers  
- Text splitters  
- Embedders  
- Retrievers  
- Chains for RAG and other use cases

The idea was to make it easier for developers to use these building blocks to build powerful LLM applications.

But this approach created **some problems**:

### ⚠️ Problems with Old Approach
1. **Heavy Codebase**  
   - Too many different components and chains made the codebase large and complex.
2. **Connection Issues Between Components**  
   - It was hard to connect different components together smoothly.
3. **Learning Overload for Developers**  
   - Developers had to learn each type of chain separately (RAG chain, LLM chain, Retrieval chain, etc.), which became confusing and slowed down development.

---

### 💡 Why Runnables Were Introduced

To solve these issues, the LangChain team introduced **Runnables**.

**Runnables are a common interface that all components can follow.**

Instead of having many different chain types with their own logic, **everything is now treated as a Runnable**.

---

## ⚙️ How Runnables Work

At the base level, `Runnable` is an **abstract class** that defines common methods like:

- `invoke(input)` – run the component on a single input
- `batch(inputs)` – run the component on multiple inputs
- `stream(input)` – stream output as it is generated

**All components (LLMs, retrievers, output parsers, etc.) now extend this `Runnable` interface.**

This means:

- **Combining two Runnables creates another Runnable.**
- **Runnables can be connected easily and consistently.**
- **Communication between different components is smooth.**

So now, **instead of learning many types of chains**, developers just need to learn **how to connect Runnables**.

---

## 📅 Day 6 – Runnable Types in LangChain

## 📌 Recap

On Day 5, we learned **why Runnables were introduced** — to give every LangChain component (LLMs, retrievers, parsers, etc.) a common interface (`invoke`, `batch`, `stream`) so they could be connected easily.

Today, we go one level deeper and look at the **different types of Runnables** LangChain gives us, and how each one solves a specific problem while building pipelines.

---

## 🧩 Two Broad Categories of Runnables

LangChain's Runnables can be split into **two categories**:

### 1️⃣ Task-Specific Runnables
These are the **core LangChain components** that have been converted into Runnables so they follow the common interface.

Examples:
- `ChatOpenAI` (LLM) → a Runnable
- `PromptTemplate` → a Runnable
- `Retriever` → a Runnable
- `OutputParser` → a Runnable

👉 In simple words: **these are the actual "workers"** — they do a specific job (call an LLM, format a prompt, fetch documents, parse output).

### 2️⃣ Runnable Primitives
These are **structural / control-flow tools**. They don't do the "work" themselves — instead, they help you **connect, arrange, and control** how the task-specific Runnables run together.

Examples:
- `RunnableSequence`
- `RunnableParallel`
- `RunnablePassthrough`
- `RunnableLambda`
- `RunnableBranch`

👉 In simple words: **these are the "glue"** — they decide the *order*, *flow*, and *logic* of how your components run.

**Analogy:** Think of task-specific Runnables as **workers on an assembly line** (one welds, one paints, one packs), and Runnable primitives as the **conveyor belt system** that decides who works after whom, who works in parallel, and who gets skipped.

---

## 1. RunnableSequence

**RunnableSequence** is a runnable primitive that runs multiple runnables **one after another**, where the output of one step becomes the input of the next step.

It's basically the `|` (pipe) operator you already use in LCEL — it chains steps in a straight line.

```
prompt  →  llm  →  output_parser
```

**Example (in words):**
1. `prompt` formats the user's topic into a proper question.
2. `llm` generates an answer.
3. `output_parser` cleans up the answer into plain text.

```python
chain = prompt | llm | output_parser
chain.invoke({"topic": "AI"})
```

Each `|` is secretly building a `RunnableSequence` behind the scenes.

---

## 2. RunnableParallel

**RunnableParallel** is a runnable primitive that allows **multiple runnables to execute in parallel**.

Each runnable receives the **same input** and processes it **independently**, producing a **dictionary of outputs**.

**Example (in words):**
Topic = "AI" is sent to two branches at once:
- Branch 1: `LLM 1` → generates a **tweet**
- Branch 2: `LLM 2` → generates a **LinkedIn post**

```
                topic = "AI"
               /            \
          LLM 1              LLM 2
            ↓                  ↓
          tweet              linkedin post
```

```python
parallel_chain = RunnableParallel({
    "tweet": tweet_chain,
    "linkedin": linkedin_chain
})

parallel_chain.invoke({"topic": "AI"})
# Output: {"tweet": "...", "linkedin": "..."}
```

**When to use it:** When you need the **same input** processed in **different ways at the same time** — like generating multiple content formats, or calling multiple models for comparison.

---

## 3. RunnablePassthrough

**RunnablePassthrough** is a runnable primitive that simply **passes the input through unchanged** — it doesn't transform anything.

**Why is this useful?** Sometimes inside a `RunnableParallel`, you want to keep the **original input** available alongside the output of another step (e.g., for a RAG pipeline where you need both the retrieved context *and* the original question).

**Example (in words):**
In a RAG chain, you want to send both:
- the **original question** (unchanged) → passthrough
- the **retrieved context** → retriever output

to the final prompt.

```python
parallel_chain = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})
```

Output:
```python
{
  "context": "<retrieved docs>",
  "question": "<the exact same question user asked>"
}
```

👉 In simple words: **it's like a "do nothing, just forward it" step** — used to preserve data that would otherwise get lost.

---

## 4. RunnableLambda

**RunnableLambda** is a runnable primitive that allows you to **apply custom Python functions** within an AI pipeline.

It acts as **middleware** between different AI components, enabling **preprocessing, transformation, API calls, filtering, and post-processing** in a LangChain workflow.

```
input  →  RunnableLambda(clean)  →  llm  →  sentiment
```

**Example (in words):**
Before sending text to the LLM, you want to **clean it** (remove extra spaces, lowercase it, strip HTML tags, etc.) using your own Python function — not a built-in LangChain component.

```python
def clean_text(text):
    return text.strip().lower()

clean_step = RunnableLambda(clean_text)

chain = clean_step | llm | sentiment_parser
```

👉 In simple words: **it turns any normal Python function into a Runnable**, so you can plug your own custom logic anywhere in the chain — not just LangChain's built-in tools.

---

## 5. RunnableBranch — condition/if chains

**RunnableBranch** is a control flow component in LangChain that allows you to **conditionally route** input data to different chains or runnables based on **custom logic**.

It functions like an **if/elif/else block for chains** — where you define a set of condition functions, each associated with a runnable (e.g., LLM call, prompt chain, or tool). The **first matching condition is executed**. If no condition matches, a **default runnable** is used (if provided).

```
                     input
              /        |          \
        complain    refund      general query
           ↓           ↓              ↓
        customer   database         chatbot
```

**Example (in words):**
A customer support pipeline routes messages based on intent:
- If message = **complaint** → send to "customer support" chain
- If message = **refund request** → send to "database lookup" chain
- Else (general query) → send to a **default chatbot** chain

```python
branch_chain = RunnableBranch(
    (lambda x: "complaint" in x["input"], complaint_chain),
    (lambda x: "refund" in x["input"], refund_chain),
    general_chatbot_chain  # default
)
```

👉 In simple words: **it's decision-making inside your pipeline** — different inputs get sent down different paths automatically.

---

## 6. LCEL (LangChain Expression Language)

**LCEL** is the **syntax/language** that lets you build all the above Runnables using simple, readable operators instead of verbose class-based code.

The most common operator is the **pipe `|`**, which chains Runnables together (this is what builds a `RunnableSequence` under the hood).

**Before LCEL (old, verbose way):**
```python
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI")
```

**With LCEL (new, clean way):**
```python
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "AI"})
```

**Why LCEL matters:**
- ✅ Cleaner, more readable pipelines
- ✅ Automatic support for `invoke`, `batch`, `stream` on the whole chain
- ✅ Easy to mix Sequence, Parallel, Passthrough, Lambda, and Branch together in one expression

**Example combining everything:**
```python
full_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | RunnableLambda(clean_text)
    | prompt
    | llm
    | output_parser
)
```

👉 In simple words: **LCEL is the "grammar" that lets you write pipelines like sentences**, combining primitives (Sequence, Parallel, Passthrough, Lambda, Branch) with task-specific Runnables (prompt, llm, parser) using simple operators.

---

## 🔑 Quick Recap Table

| Runnable | Purpose | Analogy |
|---|---|---|
| `RunnableSequence` | Run steps one after another | Assembly line |
| `RunnableParallel` | Run steps at the same time, same input | Multiple workers, same order |
| `RunnablePassthrough` | Forward input unchanged | A pass-through wire |
| `RunnableLambda` | Plug in custom Python function | Custom tool on the belt |
| `RunnableBranch` | Route input based on condition | if/elif/else for chains |
| `LCEL` | Syntax to combine all of the above | Grammar of the pipeline |

## 📅 Day 7 – RAG & Document Loaders in LangChain

## 📌 Recap

On Day 6, we learned about **Runnable Primitives** (`RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch`) and **LCEL** — the tools that help us connect components together.

Today we shift focus to a very important real-world use case: **RAG (Retrieval-Augmented Generation)**, and the first building block of any RAG pipeline — **Document Loaders**.

---

## 1. What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines **information retrieval** with **language generation** — where a model retrieves relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses.

### 🤔 Why do we even need RAG?

A normal chatbot (like plain ChatGPT) only knows what it learned during training. It struggles with two things:

```
Chatbots  →  ChatGPT
   ├──► Current affairs   (things that happened after training)
   └──► Personal data     (your own documents, company data, etc.)
```

RAG solves this by connecting the LLM to an **external knowledge base** at query time.

```
        ┌────────────┐
   ┌───►│    LLM     │─────────┐
   │    └────────────┘         ▼
   │         ▲          ┌───────────────┐
   │         │          │External       │
   │         │          │Knowledge Base │
   │         └──────────┤               │
   │                    └───────────────┘
┌──────┐
│ User │
└──────┘
```

The user asks a question → LLM checks the **external knowledge base** → relevant info is pulled back → LLM uses it to generate a grounded answer.

### ✅ Benefits of Using RAG

1. **Use of up-to-date information** – the knowledge base can be refreshed anytime, without retraining the model.
2. **Better privacy** – your private/personal documents never need to be sent to train a model; they just sit in your own knowledge base.
3. **No limit on document size** – you're not constrained by what fits in the model's training data; you can plug in as many documents as you want.

### 🏗️ What does a RAG-based application need?

A RAG pipeline is built from several components:

```
                    RAG
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼             ▼
Document Loaders  Text Splitters  Vector DBs   Retrievers
```

- **Document Loaders** → bring raw data (PDFs, text, CSVs, web pages) into LangChain
- **Text Splitters** → break large documents into smaller chunks
- **Vector Databases** → store chunk embeddings for similarity search
- **Retrievers** → fetch the most relevant chunks for a given query

Today, we focus on the first piece: **Document Loaders**.

---

## 2. What is a Document Loader & Why Do We Need It?

**Document loaders** are components in LangChain used to **load data from various sources** into a standardized format (usually as `Document` objects), which can then be used for **chunking, embedding, retrieval, and generation**.

```
pdf  ─┐
txt  ─┤
DB   ─┼──►  Common format  ──►  Document
S3   ─┘
```

Every `Document` object looks like this:

```python
Document(
    page_content="The actual text content",
    metadata={"source": "filename.pdf", ...}
)
```

### 🤔 Why do we even need this?

Data lives in **many different formats** — PDFs, plain text, databases, cloud storage, websites, CSVs. An LLM pipeline can't work directly with a raw PDF file or a raw database row.

Document Loaders act as a **universal adapter** — no matter where your data comes from, it gets converted into the same standard `Document` shape, so every downstream step (splitting, embedding, retrieval) works the same way regardless of the original source.

---

## 3. TextLoader

**TextLoader** is a simple and commonly used document loader in LangChain that reads plain text (`.txt`) files and converts them into LangChain `Document` objects.

```
.txt  ──────►  document obj
```

### Use Case
- Ideal for loading chat logs, scraped text, transcripts, code snippets, or any plain text data into a LangChain pipeline.

### ⚠️ Limitation
- Works **only** with `.txt` files.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt")
docs = loader.load()
```

---

## 4. PyPDFLoader (and other PDF loader options)

**PyPDFLoader** is a document loader in LangChain used to load content from PDF files and convert **each page** into a separate `Document` object.

```python
[
    Document(page_content="Text from page 1", metadata={"page": 0, "source": "file.pdf"}),
    Document(page_content="Text from page 2", metadata={"page": 1, "source": "file.pdf"}),
    ...
]
```

### ⚠️ Limitation
- It uses the **PyPDF** library under the hood — not great with **scanned PDFs** or **complex layouts**.

### 📚 Many More Options Exist

Since PDFs come in many flavors (clean text, tables, scanned images, complex layouts), LangChain gives you **multiple PDF loaders** to pick from:

| Use Case | Recommended Loader |
|---|---|
| Simple, clean PDFs | `PyPDFLoader` |
| PDFs with tables/columns | `PDFPlumberLoader` |
| Scanned/image PDFs | `UnstructuredPDFLoader` or `AmazonTextractPDFLoader` |
| Need layout and image data | `PyMuPDFLoader` |

👉 In simple words: **there's no "one loader fits all" for PDFs** — pick the loader based on how messy or clean your PDF is.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
docs = loader.load()
```

---

## 5. DirectoryLoader

**DirectoryLoader** is a document loader that lets you **load multiple documents from a directory (folder) of files** — instead of loading them one by one.

```
folder/
 ├── ⬜ ⬜ ⬜
 └── ⬜ ⬜ ⬜   ──►  langchain (all as Documents)
```

It uses **glob patterns** to control exactly what gets picked up:

| Glob Pattern | What It Loads |
|---|---|
| `"**/*.txt"` | All `.txt` files in all subfolders |
| `"*.pdf"` | All `.pdf` files in the root directory |
| `"data/*.csv"` | All `.csv` files in the `data/` folder |
| `"**/*"` | All files (any type, all folders) |

`**` = recursive search through subfolders.

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("my_docs/", glob="**/*.pdf")
docs = loader.load()
```

👉 In simple words: **instead of writing a loader call for every single file, point DirectoryLoader at a folder and a pattern, and it grabs everything matching that pattern for you.**

---

## 6. Load vs Lazy Load

### 🤔 The Problem

Suppose you have **4,000 or even more documents/PDFs**. If you try to load **all of them into memory at once**, that's a very heavy task — you can easily **run out of memory**, especially with large/heavy PDFs.

This is exactly where **`lazy_load()`** becomes useful.

### ✅ `load()` — Eager Loading

- Loads **everything at once**.
- Returns: a **list** of `Document` objects.
- Loads all documents **immediately** into memory.
- Best when:
  - The number of documents is small.
  - You want everything loaded upfront.

### 🔄 `lazy_load()` — Lazy Loading

- Loads **on demand**.
- Returns: a **generator** of `Document` objects.
- Documents are **not all loaded at once** — they're fetched **one at a time**, as needed.
- Best when:
  - You're dealing with **large documents or lots of files**.
  - You want to **stream** processing (e.g., chunking, embedding) without using lots of memory.

```
all pdfs  ──►  generator of docs  ──►  use one at a time (as per demand)
```

```python
# Eager
docs = loader.load()          # all 4000 docs in memory at once ❌ (risky)

# Lazy
for doc in loader.lazy_load():   # one doc at a time ✅ (memory-friendly)
    process(doc)
```

👉 In simple words: `load()` is like **downloading an entire library at once**, while `lazy_load()` is like **borrowing one book at a time as you need it**.

---

## 7. WebBaseLoader

**WebBaseLoader** is a document loader in LangChain used to **load and extract text content from web pages (URLs)**.

It uses **BeautifulSoup** under the hood to parse HTML and extract visible text.

### ✅ When to Use
- For blogs, news articles, or public websites where the content is primarily **text-based and static**.

### ⚠️ Limitations
- Doesn't handle **JavaScript-heavy pages** well (use `SeleniumURLLoader` for that).
- Loads only **static content** (what's in the raw HTML, not what loads *after* the page renders).

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/blog-post")
docs = loader.load()
```

👉 In simple words: **great for simple, static pages**; if a site depends heavily on JavaScript to render its content, this loader will miss it.

---

## 8. CSVLoader

**CSVLoader** is a document loader in LangChain used to load `.csv` files — where **each row of the CSV becomes one `Document` object**.

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("customers.csv")
docs = loader.load()
```

Each `Document` will typically look like:

```python
Document(
    page_content="name: John\nemail: john@example.com\ncity: Mumbai",
    metadata={"source": "customers.csv", "row": 0}
)
```

👉 In simple words: **think row-by-row** — one CSV row = one Document, so a 1,000-row CSV becomes 1,000 Documents.

---

## 9. Custom Document Loaders

Sometimes your data source doesn't match **any built-in loader** — maybe it's a proprietary API, a niche file format, or an internal database with a unique structure.

For these cases, **LangChain provides the facility to create your own Custom Loader**.

You do this by subclassing the base `Document Loader` interface and implementing your own `load()` (and optionally `lazy_load()`) logic:

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class MyCustomLoader(BaseLoader):
    def __init__(self, source):
        self.source = source

    def load(self):
        # your own custom logic to fetch/parse data
        raw_data = fetch_from_my_source(self.source)
        return [Document(page_content=raw_data, metadata={"source": self.source})]
```

👉 In simple words: **if none of the built-in loaders fit your requirement, you're not stuck** — LangChain lets you build your own loader that still plugs into the same pipeline as every other loader.

---

## 🔑 Quick Recap Table

| Loader | Purpose | Key Limitation |
|---|---|---|
| `TextLoader` | Load `.txt` files | Only works with `.txt` |
| `PyPDFLoader` | Load clean PDFs, page by page | Struggles with scanned/complex PDFs |
| `DirectoryLoader` | Load multiple files from a folder using glob patterns | Depends on correct glob pattern |
| `load()` | Eager loading — all at once | Memory-heavy for large datasets |
| `lazy_load()` | Lazy loading — one at a time via generator | Slightly more setup (iteration) |
| `WebBaseLoader` | Load static web pages | Fails on JS-heavy pages |
| `CSVLoader` | Load CSV, row → Document | Structure depends on CSV columns |
| Custom Loader | Handle any unsupported source | You write the logic yourself |
## Getting Started
    git clone https://github.com/SumitMARSS/LangChain_Models_Learning.git


## License
  This project is open for exploration and learning. Please credit the repository if you use its code or notes.
