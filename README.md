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



## 📅 Day 8 – Text Splitting in LangChain

## 📌 Recap

On Day 7, we learned about **RAG** and **Document Loaders** — the components that bring raw data (PDFs, text, CSVs, web pages) into LangChain as standardized `Document` objects.

But once we have these documents, there's still a problem: they can be **huge**. Today we learn about **Text Splitting** — the step that breaks large documents into smaller, LLM-friendly pieces.

---

## 1. What is Text Splitting & Why Do We Need It?

**Text Splitting** is the process of breaking large chunks of text (like articles, PDFs, HTML pages, or books) into smaller, manageable pieces (**chunks**) that an LLM can handle effectively.

```
                    ┌──────────────┐
                ┌──►│    chunk 1   │
┌────────────┐  │   └──────────────┘
│            │  │   ┌──────────────┐
│ Large Text │──┼──►│    chunk 2   │
│            │  │   └──────────────┘
└────────────┘  │   ┌──────────────┐
                └──►│    chunk 3   │
                     └──────────────┘
```

### 🤔 Why not just feed the LLM the whole large text dataset directly?

There are two big reasons:

**1. Model limitations (context length)**

Every LLM has a **maximum input size constraint**, known as its **context length**.

```
LLM  →  context length
          │
          ▼
   [50K tokens]  →  words
```

If your document is bigger than this limit, the model simply **can't accept it** in one go. Splitting allows us to process documents that would otherwise **exceed these limits**.

**2. Embedding quality drops for large text**

It's been observed that when we generate an **embedding for a large piece of text all at once**, we don't get as much accuracy/efficiency as when we split it into smaller, focused sections first.

**Example:** Suppose you have notes on the IPL (cricket league) covering all teams. If you generate **one embedding for the entire note**, the resulting vector tries to represent *everything at once* — batting stats, bowling stats, every team, every match — so it becomes a vague, "averaged out" representation.

Instead, if you split the notes **team-wise** and generate a separate embedding for each team's section, each vector stays **focused and specific**. When a user asks "How did Mumbai Indians perform?", the retriever can match against the *Mumbai Indians* embedding directly — giving a **much more accurate and relevant answer**.

```
text ──► embedding ──► vectors
```

### ✅ Where Text Splitting Helps (Downstream Tasks)

Text Splitting improves nearly every LLM-powered task:

| Task | Why Splitting Helps |
|---|---|
| **Embedding** | Short chunks yield more accurate vectors |
| **Semantic Search** | Search results point to focused info, not noise |
| **Summarization** | Prevents hallucination and topic drift |

### ⚡ Bonus: Optimizing Computational Resources

Working with smaller chunks of text can also be **more memory-efficient** and allows for **better parallelization** of processing tasks (e.g., embedding many small chunks in parallel instead of one giant blocking call).

---

## 2. Types of Text Splitters

LangChain provides **four main strategies** for splitting text:

```
                    Text Splitters
       ┌───────────────┬───────────────┬────────────────┐
       ▼               ▼               ▼                ▼
 Length Based    Text Structure   Document Structure   Semantic Meaning
                     Based              Based               Based
```

1. **Length Based** – split purely by a fixed size (characters or tokens).
2. **Text Structure Based** – split respecting natural text units (paragraphs, sentences, words).
3. **Document Structure Based** – split respecting the document's own structure (headings, code blocks, markdown sections, etc.).
4. **Semantic Meaning Based** – split based on where the *meaning* actually changes, using embeddings.

Today we go deep into the first one: **Length Based Text Splitting**.

---

## 3. Length Based Text Splitting

**Length Based Text Splitting** simply breaks the text into chunks of a fixed, predefined **size** — without worrying about sentence or paragraph boundaries.

```
Space exploration has led to incredible scientific        Space exploration has led to incredible scientific
discoveries. From landing on the Moon to exploring         discoveries. From landing on the Moon to explorin  → C1
Mars, humanity continues to push the boundaries of                                                     
what's possible beyond our planet.                       g Mars, humanity continues to push the boundaries
                                                             of what's possible beyond our planet. These missi → C2
These missions have not only expanded our knowledge
of the universe but have also contributed to               ons have not only expanded our knowledge of the
advancements in technology here on Earth. Satellite         universe but have also contributed to             → C3
communications, GPS, and even certain medical
imaging techniques trace their roots back to               n technology here on Earth. Satellite
innovations driven by space programs.                       communications, GPS, and even certain medical      → C4
                                                              techniqu

                                                             es trace their roots back to innovations driven
                                                              by space programs.                                → C5

                                          chunk size = 100 characters
```

Notice something important in the example above: chunk **C1** ends mid-word ("explorin") and **C2** starts right where it was cut off ("g Mars..."). The splitter doesn't care about words, sentences, or meaning — it **just counts and cuts** at the defined size.

### 📏 What can "length" be measured in?

```
chunks → size
        ├──► characters
        └──► tokens
```

### a) Character-Based Splitting

Splits the text every **N characters** (e.g., every 100 characters), regardless of what's in between.

### b) Token-Based Splitting

Splits the text every **N tokens** (the units an LLM actually "reads" — roughly ¾ of a word on average), which is often more aligned with how the model will actually consume the text.

### ✅ Pros
- **Easiest** to implement.
- **Fastest** — no NLP logic required, just counting.

### ❌ Cons
- Doesn't understand **vocabulary** or sentence boundaries at all.
- Doesn't care about **semantic meaning** — it just stops at the decided number of characters/tokens.
- This can **cut a sentence right in the middle**, splitting one idea across two chunks. When embeddings are generated for these two "broken" chunks, you can end up with **two different vectors that don't fully represent either half** properly — leading to **rubbish/inaccurate retrieval results**.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
chunks = splitter.split_text(large_text)
```

---

## 4. Chunk Overlapping

Since length-based splitting can cut sentences awkwardly in the middle, we use **chunk overlapping** to keep some continuity between consecutive chunks — so sentences don't lose their meaning entirely at the boundary.

**Chunk overlap** means the **end portion of one chunk is repeated at the start of the next chunk**, so context isn't abruptly lost between splits.

```python
chunk overlapping is 10 to 20% for llm models, for example if chunk size is 1000 then chunk overlap should be 100 to 200.
```

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150   # ~15% overlap, within the recommended 10-20% range
)
chunks = splitter.split_text(large_text)
```

👉 In simple words: **overlap acts like a small "bridge"** between two chunks — so if an important sentence gets cut at the boundary, part of it still appears in both chunks, helping preserve meaning during retrieval.

---

## 5. Text Structure Based Splitting

Length-based splitting has one big flaw — it can cut a sentence right in the middle. **Text Structure Based Splitting** fixes this by respecting the **natural structure of language**: paragraphs → sentences → words → characters.

Instead of blindly cutting at a fixed size, it tries to split at the **most meaningful boundary first**, and only breaks things down further if it still needs to hit the target chunk size.

### 🔄 The Divide & Merge Process

This is exactly how LangChain's **RecursiveCharacterTextSplitter** works. Let's say we want a **chunk size of 10** (just for illustration). Here's the full process, step by step:

```
Large Text
    │
    ▼
1) Try splitting by PARAGRAPH ("\n\n")
    │
    ├──► Is each piece ≤ chunk size (10)? ──► YES → keep as chunk ✅
    │
    └──► NO → too big → go one level deeper
                    │
                    ▼
        2) Try splitting by SENTENCE (". ")
                    │
                    ├──► Is each piece ≤ chunk size? ──► YES → keep as chunk ✅
                    │
                    └──► NO → still too big → go deeper
                                    │
                                    ▼
                        3) Try splitting by WORD (" ")
                                    │
                                    ├──► Is each piece ≤ chunk size? ──► YES → keep as chunk ✅
                                    │
                                    └──► NO → still too big → go deeper
                                                    │
                                                    ▼
                                        4) Split by CHARACTER
                                                    │
                                                    └──► Always fits ✅ (last resort)
```

**Divide step:** at each level, the text is broken using that level's separator (paragraph → sentence → word → character).

**Merge step:** after dividing, LangChain **greedily merges** the small pieces back together — combining as many consecutive pieces as possible — as long as their combined length still stays **within the chunk size limit**. This is what makes the final chunks as close to the target size as possible, without cutting mid-sentence unless absolutely necessary.

👉 In simple words: it **tries the "nicest" cut first (paragraph)**, and only falls back to sentence → word → character **if the text piece is still too big to fit**. This is why it's called **recursive** — it keeps recursing down to smaller units only when needed.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0
)
chunks = splitter.split_text(large_text)
```

This is why **RecursiveCharacterTextSplitter** is the **most commonly recommended default splitter** in LangChain — it gives you clean, meaning-respecting chunks in almost all everyday scenarios, without any extra setup.

---

## 6. Document Structure Based Splitting

Some documents aren't plain prose — they have their **own internal structure**. Think of:
- Markdown files (headings, code blocks, horizontal rules)
- Source code files (classes, functions)
- HTML files (tags, sections)

**Document Structure Based Splitting** uses this **built-in structure** as the natural split points — instead of paragraphs/sentences, it uses things like headings, class definitions, or function definitions.

👉 The key insight: **behind the scenes, it's using the exact same recursive divide-and-merge process as Text Structure Based Splitting** — the only thing that changes is the **list of separators**. Instead of `paragraph → sentence → word → character`, it uses separators that are meaningful for that specific document type.

### 📄 Example: Markdown Documents

```python
# First, try to split along Markdown headings (starting with level 2)
"\n#{1,6} ",
# Note the alternative syntax for headings (below) is not handled here
# Heading level 2
# --------------
# End of code block
"```\n",
# Horizontal lines
"\n\\*\\*\\*+\n",
"\n---+\n",
"\n___+\n",
# Note that this splitter doesn't handle horizontal lines defined
# by *three or more* of ***, ---, or ___, but this is not handled
"\n\n",
"\n",
" ",
"",
```

### 💻 Example: Python Code Documents

```python
# First, try to split along class definitions
"\nclass ",
"\ndef ",
"\n\tdef ",
# Now split by the normal type of lines
"\n\n",
"\n",
" ",
"",
```

**Why explicit separators are needed:** for a normal English paragraph, "sentence" and "word" boundaries are obvious (`.`, ` `). But for **code**, there's no such generic rule — you have to **explicitly tell the splitter** what counts as a meaningful boundary (`class`, `def`, indented `\tdef`, etc.), because a class or function is the "paragraph" of code.

**Example usage with a Python-aware splitter:**

```python
# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
```

Here, the splitter would first try to keep each `class`/`def` block together as one chunk (since that's the most meaningful unit of code), and only fall back to splitting by lines if a single class/function is still too large.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)
chunks = python_splitter.split_text(code_string)
```

👉 In simple words: **document structure based splitting = same recursive engine as text structure based splitting, just with a custom, format-aware separator list** (headings for Markdown, `class`/`def` for Python, tags for HTML, etc.) instead of generic paragraph/sentence separators.

---

## 7. Semantic Meaning Based Splitting

Sometimes even a **well-structured recursive split** doesn't do the job well — because it's still based purely on *structure* (paragraphs, sentences), not on *what the text is actually talking about*.

**Semantic Meaning Based Splitting** tries to understand the **meaning** of the text before splitting it, instead of just following structural rules.

### 🤔 Example: Where structure-based splitting fails

```
Farmers were working hard in the fields, preparing the soil and planting seeds for
the next season. The sun was bright, and the air smelled of earth and fresh grass.
The Indian Premier League (IPL) is the biggest cricket league in the world. People
all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates
fear in cities and villages. When such attacks happen, they leave behind pain and
sadness. To fight terrorism, we need strong laws, alert security forces, and support
from people who care about peace and safety.
```

If we split this **by paragraph** (2 paragraphs → 2 chunks), that's actually **wrong**! The first paragraph secretly contains **two completely different topics** — farming, and then IPL cricket — mixed into a single paragraph. A structure-based splitter would keep them together as **one chunk**, even though they have nothing to do with each other. What we actually want is **2 chunks** split by topic, not by paragraph:
- Chunk A: Farming
- Chunk B: IPL cricket
- Chunk C: Terrorism

```
2 paragraphs (structure)  ✗
2 chunks (by actual topic/meaning)  ✓
```

### 🔄 How It Works — The Sliding Window Approach

1. **Break everything into sentences** (across the whole text, ignoring paragraph boundaries).
2. **Generate an embedding for each sentence.**
3. **Slide through the sentences one by one**, and compare the embedding similarity between **each sentence and the next sentence**.
4. If two consecutive sentences have **high similarity** → they're likely talking about the same topic → keep them in the **same chunk**.
5. If two consecutive sentences have **very low similarity** → that's a signal the **topic has changed** → **split here**, starting a new chunk.

```
sentence 1  ──similarity──  sentence 2  ──similarity──  sentence 3  ──similarity──  sentence 4
   (high)                      (low, topic changes)          (high)
      └── same chunk ──┘            └── new chunk starts ─────────────┘
```

This works on a simple principle: **if sentences talk about the same topic, they should have high semantic similarity; when the topic shifts, similarity drops sharply.**

### ⚠️ Notes & Limitations
- This approach is currently in an **experimental stage** in LangChain.
- **"How low is low enough" to trigger a split** isn't a single fixed number — it depends on different criteria/thresholding methods, such as:
  - Standard deviation of similarity scores
  - Percentile-based thresholds
  - Interquartile range, and other statistical methods

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation"
)
chunks = splitter.split_text(large_text)
```

👉 In simple words: **it lets the meaning of the text decide where to split**, not the punctuation or paragraph breaks — making it the most "intelligent" splitter of the four, at the cost of being newer/less battle-tested.

---

## 🔑 Quick Recap Table

| Concept | Key Idea |
|---|---|
| Text Splitting | Break large text into smaller chunks an LLM can handle |
| Why needed | Model context length limits + better embedding accuracy |
| Types of splitters | Length based, Text structure based, Document structure based, Semantic meaning based |
| Length Based Splitting | Splits by fixed char/token count — fast but ignores meaning |
| Character vs Token | Two ways to measure "length" for splitting |
| Chunk Overlap | Repeats 10-20% of previous chunk at the start of next, to preserve context |
| Text Structure Based | Recursive divide & merge: paragraph → sentence → word → character |
| Document Structure Based | Same recursive engine, but with format-aware separators (headings, `class`/`def`, tags) |
| Semantic Meaning Based | Sliding window over sentence embeddings; splits where similarity drops sharply (experimental) |


## 📅 Day 9 – Vector Databases

## 📌 Recap

On Day 8, we learned about **Text Splitting** — breaking large documents into smaller, meaningful chunks. Once we have these chunks, the next step in a RAG pipeline is to convert them into **embeddings** and store them somewhere we can search efficiently. That "somewhere" is a **Vector Database** — today's topic.

---

## 1. The Problem: Movie Recommendations

Let's start with a real-world story.

Suppose you're building a **movie recommendation system**. When someone is on the page for **Spider-Man**, you want to recommend similar movies — like **Iron Man**, or other Marvel movies — so the user keeps interacting with your website.

**Big question: how do we decide what's "similar"?**

### ❌ Attempt 1: Keyword Matching

The obvious first idea is to match **keywords** — same actor, same director, same genre tag, etc.

But this approach has serious flaws:

1. **No shared keywords → similar movies never get discovered.** If two movies are genuinely similar in spirit/theme but don't share an actor, director, or tag, the system completely misses the connection.
2. **Same keywords, different feel.** Some movies share keywords (e.g., same actor) but belong to a **completely different genre** — a comedy and a thriller with the same lead actor aren't "similar" just because of that overlap.

### ✅ Attempt 2: Match by Plot (Meaning, Not Keywords)

A much better approach: match movies based on their **plot/description** — i.e., their actual **meaning** — instead of surface-level keywords.

Here's how:
1. Take the **plot/description** of every movie in your catalog.
2. Generate an **embedding** (a vector) for each movie's plot.
3. When a user is viewing a movie, take **that movie's plot embedding** and compare it against **all other movie embeddings**.
4. Recommend the movies whose embeddings are **most similar** (closest in vector space).

To do this at scale, we need somewhere to **store all these movie vectors** — and that's exactly why we need a **Vector Database**.

---

## 2. What is a Vector Store?

**A vector store is a system designed to store and retrieve data represented as numerical vectors (embeddings).**

### 🔑 Key Features

1. **Storage** – Ensures that vectors and their associated metadata are retained, whether:
   - **in-memory** for quick lookups, or
   - **on-disk** (hard drive) for durability and large-scale use.
2. **Similarity Search** – Helps retrieve the vectors most similar to a query vector.
3. **Indexing** – Provides a data structure or method that enables fast similarity searches on high-dimensional vectors (e.g., approximate nearest neighbor lookups).
4. **CRUD Operations** – Manages the lifecycle of data: adding new vectors, reading them, updating existing entries, and removing outdated vectors.

```
                     ┌──────────────┐
                ┌───►│  in-memory   │
   vector store─┤    └──────────────┘
                └───►│  on-disk     │──► hard drive
                     └──────────────┘
```

### 📌 Use Cases
1. Semantic Search
2. RAG
3. Recommender Systems (exactly our movie example!)
4. Image/Multimedia Search

---

## 3. Understanding Embeddings & Storage (in Simple Words)

Every movie plot gets converted into a **vector** — basically just a **long list of numbers**. Each number captures some tiny aspect of "meaning." The **length of this list** is called the **embedding dimension**.

```
        ┌─────────────────────────┐
   v_q ─┤  0.12  0.98  -0.44  ...  │   ← 768 numbers (if 768-dim)
        └─────────────────────────┘
              768-dim
```

Think of it like coordinates: a 2-dimensional vector is a point on a flat map `(x, y)`. A 768-dimensional vector is a point in a space with 768 "directions" — impossible to picture, but the computer can still measure **distance/similarity** between two such points just fine.

### 📊 Common Embedding Dimensions

| Dimension | Examples | Pros | Cons |
|---|---|---|---|
| 256 | Small custom models | Very fast, low storage | Lower semantic quality |
| 384 | MiniLM, all-MiniLM-L6-v2 | Fast, lightweight | Good for small/medium datasets |
| 512 | Some E5/GTE models | Balanced | Less expressive than larger models |
| 768 | BERT, BGE-base, E5-base | Excellent balance | More storage and compute |
| 1024 | BGE-large, E5-large | Better semantic understanding | Larger index |
| 1536 | OpenAI `text-embedding-3-small` | High quality | Larger vectors |
| 3072 | OpenAI `text-embedding-3-large` | Excellent retrieval quality | Highest storage and compute cost |

**Generally speaking:**
- **Higher dimensions** → capture more semantic info, often improve retrieval quality, but need more storage/memory and cost more to index/search.
- **Lower dimensions** → faster to compute and search, smaller vector databases, but may lose some semantic nuance.

### 🤔 Why can a 768-dimensional model sometimes outperform a 3072-dimensional model?

This comes down to one important idea: **bigger isn't always better.**

A model's quality depends not just on **how many numbers** it uses to represent meaning, but on **how well it was trained** and **how well-suited it is to your specific data**. A **well-trained 768-dimensional model** has already learned to pack meaningful, high-quality semantic information into a smaller space efficiently. A **poorly-trained (or mismatched) 3072-dimensional model**, on the other hand, might have *more room* to store information but doesn't actually use that room well — so its extra dimensions end up being mostly redundant or noisy for your use case.

👉 In simple words: **dimension size is like the size of a suitcase, but training quality is how well you pack it.** A smaller, well-packed suitcase (768-dim, well-trained) can carry more useful stuff than a huge, messily-packed one (3072-dim, poorly trained or unsuited to your data).

### 💾 Storage Example

Suppose you have **1 million documents/movies**, and you store embeddings as 32-bit floats (4 bytes each):

| Dimension | Storage per Vector | Total Storage (1M vectors) |
|---|---|---|
| 384 | ~1.5 KB | ~1.5 GB |
| 768 | ~3 KB | ~3 GB |
| 1536 | ~6 KB | ~6 GB |
| 3072 | ~12 KB | ~12 GB |

*(Actual storage may vary slightly depending on the vector database and indexing method.)*

### 🤔 Which dimension should you choose?
- **384**: Fast prototypes, chatbots, and smaller RAG systems.
- **768**: A great default for many production applications.
- **1024–1536**: High-quality enterprise search and larger RAG systems.
- **3072**: Use when maximizing retrieval quality is more important than storage and latency.

---

## 4. The Scaling Problem: 1 Crore Movies 😱

Now here's a new problem. Suppose your movie catalog has **1 crore (10 million) movies**, not just a handful.

If a user is viewing one movie, and we want to find the most similar ones, the naive approach is:

> Compare this movie's vector against **every single one** of the 1 crore other movie vectors, one by one.

This is called a **linear/brute-force search**, and its cost grows directly with the number of vectors — **O(n)**. With 1 crore vectors, this becomes **extremely slow and computationally expensive** to do for every single user, every single time.

### 🧩 So we really have 3 challenges to solve:
1. **Generate embeddings** for all movies.
2. **Store** these embeddings efficiently.
3. **Match/find similarity** quickly — without comparing against all 1 crore vectors every time.

**Challenge #3 is solved by Vector Databases using a technique called Indexing.**

---

## 5. Indexing — How Vector Databases Search Smartly

**Indexing** provides a data structure or method that enables **fast similarity search** on high-dimensional vectors — instead of a slow, brute-force, one-by-one comparison.

### 🔄 A Common Technique: Clustering

One of the most intuitive indexing strategies is **clustering**. Here's how it works, step by step:

**Step 1 — Group similar vectors into clusters.**

Take all your vectors (say, 1 crore movie vectors) and group them into a smaller number of **clusters** (say, 10 clusters) — where each cluster contains vectors that are "close" to one another.

```
                     ┌────────────────────┐
                     │ 1 crore movie       │
                     │ vectors             │
                     └──────────┬──────────┘
                                ▼
                          clustering
                                ▼
                     ┌────────────────────┐
                     │   10 clusters       │
                     │  (1 crore ÷ 10)     │
                     └────────────────────┘
```

**Step 2 — Find the centroid of each cluster.**

For every cluster, calculate its **centroid** — essentially the "average vector," representing the center point of that cluster.

```
avg → [ 10 centroid vectors ]   ← just 10 vectors now, one per cluster
```

**Step 3 — At query time, compare against centroids first, not all vectors.**

When a new query vector (`v_q`) comes in:
1. Compare it against just the **10 centroids** (not all 1 crore vectors!) → find the **closest centroid**.
2. Once you know which **cluster** the query vector is most similar to, you only need to search **inside that one cluster** for the actual best-matching vectors.

```
query vector
     │
     ▼
compare with 10 centroids  ──►  pick closest centroid  ──►  search only within that cluster
                                                                (instead of all 1 crore vectors)
```

**Why this is so much faster:** instead of comparing your query against **1 crore vectors**, you now compare against **10 centroids** + a much smaller subset of vectors **inside one cluster**. This massively cuts down the search space, turning an expensive brute-force search into a fast, approximate one.

👉 In simple words: **it's like searching for a book in a library.** Instead of checking every single book on every shelf (brute force), you first figure out which **section** (cluster) your book belongs to by checking the section signs (centroids), then you only search **within that one section**.

---

## 6. Vector Store vs Vector Database

These terms are often used interchangeably, but there's an important distinction.

```
                    ┌───────────────┐
                    │  Vector Store  │
                    │    System      │
                    └───────┬───────┘
                    ┌───────┴────────┐
                    ▼                ▼
                Storage          Retrieval
```

### 📦 Vector Store
- Typically refers to a **lightweight library or service** that focuses on **storing vectors (embeddings)** and **performing similarity search**.
- May **not** include many traditional database features like transactions, rich query languages, or role-based access control.
- Ideal for **prototyping**, smaller-scale applications.
- Example: **FAISS** — where you store vectors and can query them by similarity, but you handle persistence and scaling separately.

### 🗄️ Vector Database
- A **full-fledged database system** designed to store and query vectors.
- Offers additional **"database-like" features**:
  - Distributed architecture for horizontal scaling
  - Durability and persistence (replication, backup/restore)
  - Metadata handling (schemas, filters)
  - Potential for ACID or near-ACID guarantees
  - Authentication/authorization and more advanced security
- Geared for **production environments** with significant scaling, large datasets.

```
                    ┌───────────────────────────┐
                    │      Vector Database        │
                    │  ┌─────────────────────┐    │
                    │  │     Vector Store      │   │
                    │  │  (storage + search)   │   │
                    │  └─────────────────────┘    │
                    │  + Distributed architecture  │
                    │  + Backup and restore        │
                    │  + ACID transactions          │
                    │  + Concurrency                │
                    │  + Authentication              │
                    └───────────────────────────┘
```

👉 In simple words: **a Vector Database is effectively a Vector Store *plus* database-grade features** (clustering/indexing at scale, backup, security, concurrency, etc.). So **Vector Database is the bigger circle**, and **Vector Store is a smaller circle inside it** — every vector database contains vector-store-like capabilities, but not every vector store has full database features.

---

## 7. Vector Stores in LangChain

LangChain gives you a **common interface** so you don't get locked into one specific vector store or database.

- **Supported Stores**: LangChain integrates with multiple vector stores (**FAISS, Pinecone, Chroma, Qdrant, Weaviate**, etc.), giving you flexibility in scale, features, and deployment.
- **Common Interface**: A uniform Vector Store API lets you swap out one backend (e.g., FAISS) for another (e.g., Pinecone) with **minimal code changes**.
- **Metadata Handling**: Most vector stores in LangChain allow you to attach metadata (e.g., timestamps, authors) to each document, enabling filter-based retrieval.

```python
# Common methods across (almost) any vector store in LangChain:

from_documents(...)   or   from_texts(...)
add_documents(...)    or   add_texts(...)
similarity_search(query, k=...)

# Plus: Metadata-Based Filtering
```

### 🔁 Why this matters — swapping backends easily

```
VS  →  FAISS   (crossed out — swapping)
   →  Pinecone  (switched to)
```

```python
# Using FAISS
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)

# Switching to Pinecone — same shape of code, different class
from langchain_community.vectorstores import Pinecone
vectorstore = Pinecone.from_documents(docs, embedding_model, index_name="movies")

# Everything downstream stays the same:
results = vectorstore.similarity_search("spider man plot", k=5)
```

👉 In simple words: **LangChain built this interface specifically so that if you ever need to switch your vector store/database, you don't rewrite your whole pipeline** — you mostly just change a config/class name, and the rest of your code (`similarity_search`, `add_documents`, etc.) keeps working exactly the same. LangChain gives you that portability/authority over your backend choice.

---

## 8. Chroma — A Lightweight Vector Database

**Chroma is a lightweight, open-source vector database** that is especially friendly for **local development** and **small- to medium-scale production needs**.

It sits somewhere **between** a pure Vector Store and a full-scale Vector Database — it's **open-source** and has database-like organization, but it's **lightweight**, so it doesn't carry the full weight/complexity of a large enterprise-grade vector database.

```
[ Vector Store ]                    [ Vector DB ]
      (small,                        (full-scale,
   lightweight)  ◄────  Chroma  ────►  heavier)
```

### 🏗️ Chroma Tenancy and DB Hierarchy

```
                    user
                     │
                     ▼
                  Tenant
                 /       \
         Database         Database
          /    \            /    \
   Collection Collection  Collection Collection   ← like a "Table"
      /  \        /  \        /  \        /  \
   Doc  Doc    Doc  Doc    Doc  Doc    Doc  Doc
    │
    ├── embedding
    └── metadata
```

- **Tenant** → represents a user/organization (top-level isolation).
- **Database** → a logical grouping under a tenant.
- **Collection** → similar to a **"table"** in a traditional database — groups related documents together.
- **Doc** → an individual entry, storing its **embedding** and **metadata**.

👉 In simple words: **Chroma organizes your vectors the way a normal database organizes tables and rows** — user → database → collection ("table") → documents (each with an embedding + metadata) — while staying simple enough to run locally with minimal setup.

---

## 🔑 Quick Recap Table

| Concept | Key Idea |
|---|---|
| Problem | Keyword matching fails to find truly similar movies |
| Solution | Embed movie plots → compare vectors by similarity |
| Embedding Dimension | More numbers per vector = more storage, not always more accuracy |
| 768 vs 3072 | A well-trained smaller model can beat a poorly-trained larger one |
| Scaling problem | Comparing a query against millions of vectors (O(n)) is too slow |
| Indexing (Clustering) | Group vectors into clusters, compare against centroids first, then search within the best cluster |
| Vector Store | Lightweight storage + similarity search (e.g., FAISS) |
| Vector Database | Vector Store + database features (scaling, backup, ACID, security) — the bigger circle |
| LangChain Interface | Common API (`from_documents`, `similarity_search`, etc.) — swap backends with minimal code change |
| Chroma | Lightweight, open-source vector database — sits between Vector Store and full Vector Database |

## 📅 Day 10 – Retrievers in LangChain

## 📌 Recap

On Day 9, we learned about **Vector Databases** — how movie/document embeddings get stored, indexed, and searched efficiently using techniques like clustering.

Today we look at the component that actually **uses** that vector database (or any other data source) to fetch relevant information for a user's query: the **Retriever**.

---

## 1. What Are Retrievers?

**A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user's query.**

```
                     ┌──────────────┐
   Query   ────────► │  Retriever   │ ────────►  Documents
  (diamond)           └──────────────┘
```

Two important things to remember:
- There are **multiple types of retrievers**.
- **All retrievers in LangChain are Runnables** — meaning every retriever supports `invoke()`, `batch()`, and `stream()`, and can be plugged directly into an LCEL chain, just like the Runnables we studied on Day 6.

---

## 2. Types of Retrievers — Two Broad Categories

We can distinguish retrievers into **two categories**:

1. **Based on Data Source** — how/where the retriever fetches documents *from*.
2. **Based on Search Strategy** — *how cleverly* the retriever searches once it has a data source.

```
                          Retriever
                    ┌──────────────────────┐
                    ▼                       ▼
             Data Source              Search Strategy
                    │                       │
       ┌────────────┼────────────┐    ┌─────┼──────────────┐
       ▼            ▼             ▼    ▼     ▼              ▼
 Wikipedia      Vector Store   Arxiv  MMR  Multi-Query   Contextual
 Retriever                    Retriever                  Compression
```

- **Based on Data Source**: Wikipedia Retriever, Vector Store Retriever, Arxiv Retriever, etc. — these differ in **where** they go to fetch documents.
- **Based on Search Strategy**: MMR, Multi-Query Retriever, Contextual Compression Retriever, etc. — these differ in **how** they search/refine results, often regardless of the underlying data source.

Let's go through each one.

---

## 3. Wikipedia Retriever *(Data Source Based)*

**A Wikipedia Retriever is a retriever that queries the Wikipedia API to fetch relevant content for a given query.**

Unlike a plain document loader (which just loads whatever you point it at), a retriever applies some **intelligence/technique** to fetch only the **relevant** documents rather than loading an entire page or article blindly.

```
"ipl"  ────►  [ Wikipedia API ]  ────►  doc
```

### ⚙️ How It Works
1. You give it a query (e.g., "Albert Einstein").
2. It sends the query to Wikipedia's API.
3. It retrieves the **most relevant articles**.
4. It returns them as LangChain `Document` objects.

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever()
docs = retriever.invoke("Albert Einstein")
```

---

## 4. Vector Store Retriever *(Data Source Based)*

**A Vector Store Retriever in LangChain is the most common type of retriever** that lets you search and fetch documents from a vector store based on **semantic similarity** using vector embeddings.

### ⚙️ How It Works
1. You store your documents in a **vector store** (like FAISS, Chroma, Weaviate).
2. Each document is converted into a **dense vector** using an **embedding model**.
3. When the user enters a query:
   - It's also turned into a **vector**.
   - The retriever compares the query vector with the stored vectors.
   - It retrieves the **top-k most similar ones**.

```python
vectorstore = Chroma.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

results = retriever.invoke("What is photosynthesis?")
```

This is essentially the retriever "front door" to everything we learned about vector databases and indexing on Day 9.

---

## 5. Maximal Marginal Relevance (MMR) *(Search Strategy Based)*

> *"How can we pick results that are not only relevant to the query but also different from each other?"*

**MMR is an information retrieval algorithm designed to reduce redundancy in the retrieved results while maintaining high relevance to the query.**

### 🤔 Why MMR Retriever?

In a regular similarity search, you may get documents that are:
- All very similar to each other
- Repeating the same info
- Lacking diverse perspectives

**MMR Retriever avoids that by:**
- Picking the **most relevant document** first.
- Then picking the next most relevant document **that is also least similar to the ones already selected**.
- And so on...

This helps especially in **RAG pipelines**, where you want your context window to contain **diverse but still relevant information** — this is especially useful when documents are semantically overlapping.

### 📊 Example

| Doc ID | Content |
|---|---|
| D1 | "Climate change is causing glaciers to melt rapidly in the Arctic region." |
| D2 | "Glaciers in the Arctic are melting at an alarming rate due to rising temperatures." |
| D3 | "Deforestation in the Amazon is accelerating global climate change." |
| D4 | "Climate change is increasing the frequency of wildfires in California." |
| D5 | "Rising sea levels due to climate change threaten coastal cities like Mumbai and New York." |

**❌ Regular Similarity Search — Top 3 results:**
```
1. D1: Arctic glaciers melting
2. D2: Arctic glaciers melting     ← basically repeats D1
3. D3: Deforestation in Amazon
```

**✅ MMR — Top 3 results (diverse perspectives):**
```
1. D1: Arctic glaciers melting
2. D4: Wildfires in California
3. D5: Rising sea levels in coastal cities
```

Notice how MMR avoids picking D2 right after D1 (since they say almost the same thing), and instead brings in **different angles** of the same broader topic (climate change) — glaciers, wildfires, and sea levels — giving the LLM a much richer, less repetitive context to work with.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)
```

---

## 6. Multi-Query Retriever *(Search Strategy Based)*

**Sometimes a single query might not capture all the ways information is phrased in your documents.**

### 🤔 Example

Query: **"How can I stay healthy?"**

This could actually mean several different things:
- What should I eat?
- How often should I exercise?
- How can I manage stress?

A **simple similarity search might miss documents** that talk about those things but don't literally use the word "healthy."

### 💡 The Idea

The **Multi-Query Retriever** tries to reduce this **ambiguity** in the user's question by generating **multiple different (but semantically related) queries** based on the original, ambiguous question — and then searching with all of them.

```
query ──► vector store   (single query — might miss relevant docs)
```

becomes:

```
                    "How can I stay healthy?"
                              │
                              ▼
                            LLM
              ┌───────┬───────┼───────┬───────┐
              ▼       ▼       ▼       ▼       ▼
             q1      q2      q3      q4      q5
"What are the    "How often    "What lifestyle   "How can I     "What daily
best foods to    should I      habits improve     boost my       routines
maintain good    exercise to   mental and         immune         support
health?"         stay fit?"    physical           system         long-term
                                wellness?"         naturally?"    health?"
              │       │       │       │       │
              ▼       ▼       ▼       ▼       ▼
          (each sub-query retrieves its own documents)
```

### ⚙️ How It Works
1. Takes your **original query**.
2. Uses an **LLM** (e.g., GPT-3.5) to generate **multiple semantically different versions** of that query.
3. **Performs retrieval for each sub-query.**
4. **Combines and deduplicates** the results.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
docs = retriever.invoke("How can I stay healthy?")
```

👉 In simple words: **instead of relying on one exact phrasing of the question, it asks the same question in several different ways** — so it can find relevant documents even if they don't use the user's exact wording. The final answer is better because the question is now being **properly understood from multiple angles**, not just taken at face value.

---

## 7. Contextual Compression Retriever *(Search Strategy Based)*

**The Contextual Compression Retriever in LangChain is an advanced retriever that improves retrieval quality by compressing documents *after* retrieval — keeping only the content relevant to the user's query.**

### ❓ Example Problem

Query: *"What is photosynthesis?"*

**Retrieved Document (by a traditional retriever):**
```
"The Grand Canyon is a famous natural site.
Photosynthesis is how plants convert light into energy.
Many tourists visit every year."
```

### ❌ Problem
- The retriever returns the **entire paragraph**.
- Only **one sentence** is actually relevant to the query.
- The rest is **irrelevant noise** that wastes the context window and may confuse the LLM.

### ✅ What Contextual Compression Retriever Does

It returns only the relevant part, e.g.:
```
"Photosynthesis is how plants convert light into energy."
```

### 🔄 How the Process Works (K = 2 example)

Say the base retriever fetches `k = 2` documents (D1 and D2) for a query:

```
                                            D1 ──► LLM ──► D1 (compressed)
                             ┌───► D1  ────┘
query ──► retrieval ─────────┤
                             └───► D2  ────┐
                                            D2 ──► LLM ──► D2 (compressed)
```

For **each retrieved document**, we send a pair — `(query, document)` — to an **LLM**, and ask it to **trim out anything irrelevant** to the query, keeping only what actually answers it. This is repeated for every retrieved document.

### ⚙️ How It Works
1. **Base Retriever** (e.g., FAISS, Chroma) retrieves **N documents**.
2. A **compressor** (usually an LLM) is applied to **each document**.
3. The compressor keeps **only the parts relevant to the query**.
4. **Irrelevant content is discarded.**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

docs = compression_retriever.invoke("What is photosynthesis?")
```

### ✅ When to Use
- Your documents are **long and contain mixed information**.
- You want to **reduce context length** for LLMs.
- You need to **improve answer accuracy** in RAG pipelines.

---

## 8. And There's More...

The retrievers covered here — Wikipedia, Vector Store, Arxiv (data-source based), and MMR, Multi-Query, Contextual Compression (search-strategy based) — are only some of the most commonly used ones.

**LangChain's documentation** lists many more specialized retrievers for different use cases (self-query retrievers, ensemble retrievers, parent-document retrievers, and more). It's always worth checking the official docs when a use case doesn't quite fit the ones covered here.

### 🤔 Why do we even need this many retrievers?

Because **a plain, simple retriever doesn't perform well enough in real RAG pipelines** on its own — a single similarity search can return redundant, irrelevant, or overly literal results. So, to get genuinely good answers out of a RAG system, we reach for **advanced retrievers** — each solving a specific weakness (redundancy, ambiguity, noisy documents, etc.) — to sharpen what actually reaches the LLM.

---

## 🔑 Quick Recap Table

| Retriever | Category | Solves |
|---|---|---|
| Wikipedia Retriever | Data Source | Fetches relevant Wikipedia content via API |
| Vector Store Retriever | Data Source | Semantic similarity search over embedded documents |
| Arxiv Retriever | Data Source | Fetches relevant research papers |
| MMR | Search Strategy | Reduces redundancy, increases diversity in results |
| Multi-Query Retriever | Search Strategy | Reduces ambiguity by generating multiple query variations |
| Contextual Compression Retriever | Search Strategy | Trims retrieved documents to only the relevant parts |

## 📅 Day 11 – Why RAG? (From LLM Limitations to Retrieval-Augmented Generation)

## 📌 Recap

From Day 6 to Day 10, we built up the individual pieces of a RAG pipeline: Runnables & LCEL, Document Loaders, Text Splitters, Vector Databases, and Retrievers.

Today we zoom out and answer the bigger question: **why does RAG exist in the first place?** We trace the full story — from how an LLM "knows" things, to why that knowledge breaks down, to how RAG fixes it.

> 📝 **A quick note on your Day 11 draft**: your understanding is correct throughout — nice work. Just a couple of small terminology cleanups baked into this doc: it's **hallucination** (not "haliculation"), and **RLHF** stands for **Reinforcement Learning from Human Feedback**. Everything else — the 3 problems, the fine-tuning types, in-context learning as an emergent property, and the 4 RAG steps — is accurate.

---

## 1. What is an LLM, and Where Does Its Knowledge Come From?

An LLM's "knowledge" isn't stored as text or facts in a database — it's stored as **parameters** (just numbers/weights) inside the model itself, learned during training.

```
                    ┌────────────────────┐
   Query  ────────► │        LLM          │ ────────► Response
  (Prompt)           │  (parameters/weights) │
                    └────────────────────┘
                       [ Parametric Knowledge ]
```

The general rule: **the bigger and better-trained the parameter set, the more capable the LLM.** This is exactly why an LLM's knowledge is called **parametric knowledge** — it's knowledge baked directly into the model's parameters, not looked up from an external source.

### ⚠️ Hallucination

**Hallucination** is when an LLM provides a **factually wrong answer with confidence** — stating something incorrect as if it were certain fact, with no indication it might be wrong.

### 🧩 3 Core Problems With Relying Only on Parametric Knowledge

1. **Private data** – the LLM never saw your company's internal documents, personal files, or proprietary data during training, so it simply doesn't know about them.
2. **Recent data** – anything that happened after the model's training cutoff is unknown to it.
3. **Hallucination** – even for things it "should" know, it can confidently state incorrect information.

---

## 2. First Attempted Fix: Fine-Tuning

The natural first idea to fix these problems is: **just retrain the model on the missing/updated data.** This is called **fine-tuning**.

```
pretrained LLM  ──────►  smaller, domain-specific model
```

You take a large, general-purpose **pretrained** LLM and further train it — usually on a **smaller, curated, domain-specific** dataset — so it becomes specialized for a narrower use case.

### 🔧 Types of Fine-Tuning

```
                    Fine-Tuning
       ┌────────────────┬─────────────────────┬────────────┐
       ▼                ▼                      ▼            
  Supervised      Continued Pretraining      RLHF
  Fine-Tuning
```

**a) Supervised Fine-Tuning (SFT)**
Uses **labeled data** — pairs of `prompt → desired output`. The model is explicitly shown "for this input, this is the correct output" and learns to reproduce that mapping.

```
prompt  ──────►  desired output
```

**b) Continued Pretraining**
An **unsupervised** approach — the model continues training on a large body of **unlabeled domain text/transcripts** (no explicit prompt → output pairs), simply absorbing patterns and knowledge from that text, the same way it was originally pretrained, just on new/narrower data.

```
continued pretraining  ──►  unsupervised  ──►  transcript  ──►  model
```

**c) RLHF (Reinforcement Learning from Human Feedback)**
The model's outputs are ranked/rated by **human feedback**, and that feedback signal is used to further tune the model's behavior — pushing it toward outputs humans actually prefer.

### 🛠️ A Typical Fine-Tuning Workflow

| Step | What Happens |
|---|---|
| 1. Collect data | A few hundred to a few hundred-thousand carefully curated examples (prompts → desired outputs). |
| 2. Choose a method | Full-parameter FT, LoRA/QLoRA, or parameter-efficient adapters. |
| 3. Train for a few epochs | Keep the base weights frozen or partially frozen and update only a small subset (LoRA) **or** all weights (full FT). |
| 4. Evaluate & safety-test | Measure exact-match, factuality, and hallucination rate against held-out data; red-team for safety. |

### ✅ Could This Solve Our 3 Problems?

In theory, yes — fine-tuning can help. For example, for **hallucination**, you could fine-tune the model so that for a certain class of complex/uncertain queries, it learns to say "I don't know" or refuse to answer, rather than confidently guessing.

### ❌ But There's a Big Concern

1. **Expensive** – training even a "small" fine-tuning job on a big base LLM requires significant compute — this is costly.
2. **Needs specialized skills** – you need a proper AI/ML engineer to curate data, choose a method, train, and evaluate safely.
3. **Doesn't scale with frequently changing data** – if your data is being added/updated **frequently** (e.g., daily news, live company data), you simply **can't re-fine-tune the model every time** something changes. Fine-tuning is a slow, periodic, expensive process — not a real-time update mechanism.

👉 In simple words: fine-tuning **can** patch specific problems, but it's the wrong tool for **fast-changing or private, ever-growing data** — it's too slow, too costly, and too specialized to run continuously.

---

## 3. Second Idea: In-Context Learning

Here's an interesting observation about **humans**: we can often perform a new task just by seeing **a few examples**, or by being given **simple instructions** — without needing to be "retrained" from scratch.

> *"Humans can generally perform a new language task from only a few examples or from simple instructions — something which current NLP systems still largely struggle to do."* — from the GPT-3 paper ("Language Models are Few-Shot Learners")

The question this inspired: **can LLMs do the same thing?**

It turns out — **yes**, but only once models got large enough. Very large LLMs (like GPT-3, with 175 billion parameters) started showing strong **few-shot performance**: given just a handful of examples directly in the prompt, the model could generalize to the pattern **without any weight updates or fine-tuning at all**.

### 💡 In-Context Learning (ICL)

**In-Context Learning is a core capability of Large Language Models (LLMs)** — like GPT-3/4, Claude, and Llama — where the model learns to solve a task purely by **seeing examples in the prompt**, without updating its weights.

**Example prompt:**
```
Below are examples of texts labeled with sentiment.
Use these examples to determine the sentiment of new text.

Text: I love this phone. It's so smooth. → Positive
Text: This app crashes a lot. → Negative
Text: The camera is amazing! → Positive

Text: I hate the battery life. → ?
```

The model has never been "trained" specifically on this labeling task — it simply infers the pattern from the examples shown **right there in the prompt**, and applies it to the new input.

### 🌱 Emergent Properties

This ability is called an **emergent property**.

**An emergent property is a behaviour or ability that suddenly appears in a system when it reaches a certain scale or complexity** — even though it was not explicitly programmed or expected from the individual components.

```
LLM  ──►  emergent property  ──►  in-context learning
```

👉 In simple words: **nobody explicitly taught GPT-3 how to do in-context learning** — it simply *emerged* once the model became large enough. Smaller models don't reliably show this ability; it appears only past a certain scale.

---

## 4. From In-Context Learning to RAG

In-context learning works great with a few example **input → output** pairs. But we can push this idea further: **instead of just giving the model examples, what if we give it *extra factual context/information* relevant to the question — right there in the prompt?**

**RAG is a way to make a language model (like ChatGPT) smarter by giving it extra information at the time you ask your question.**

```
   Query ───────┐
                ▼
              Prompt ──────► LLM ──────► Response
   Context ─────┘
```

Instead of the model relying *only* on what it memorized during training (parametric knowledge), we now feed it fresh, specific, relevant information **at query time** — solving exactly the 3 problems fine-tuning struggled with: private data, recent data, and (partially) hallucination — without ever touching the model's weights.

---

## 5. The 4 Steps of RAG

**RAG contains 4 steps:**

```
   ┌────────────┐    ┌────────────┐    ┌───────────────┐    ┌────────────┐
   │  Indexing   │───►│ Retrieval   │───►│ Augmentation   │───►│ Generation  │
   └────────────┘    └────────────┘    └───────────────┘    └────────────┘
```

1. **Indexing** – We build and maintain a **knowledge base**, from which we'll later provide context to the LLM.
2. **Retrieval** – We try to find the **most relevant context** as per the user's query, from that knowledge base.
3. **Augmentation** – We give the LLM the **most relevant context, along with the user's query** — combining both into a single prompt.
4. **Generation** – The model **generates the answer**, based on the prompt created during augmentation.

Let's zoom into the first step in detail — the one that connects directly back to everything we studied on Days 7–9.

---

## 6. Indexing — Building the Knowledge Base

**Indexing** is where we prepare all our raw data so it can later be searched efficiently. It has **4 sub-steps** — and you'll recognize every single one from earlier days:

```
   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐   ┌────────────────────┐
   │  Document     │──►│  Text Chunking  │──►│  Embedding         │──►│  Storage in         │
   │  Ingestion    │   │                │   │  Generation        │   │  Vector Store       │
   └──────────────┘   └────────────────┘   └──────────────────┘   └────────────────────┘
   (Day 7)              (Day 8)              (Day 9)                 (Day 9)
```

1. **Document Ingestion** – Loading raw data (PDFs, text, CSVs, web pages, etc.) using **Document Loaders** *(Day 7)*.
2. **Text Chunking** – Breaking large documents into smaller, meaningful pieces using **Text Splitters** *(Day 8)*.
3. **Embedding Generation** – Converting each chunk into a numerical **vector** using an embedding model *(Day 9)*.
4. **Storage in Vector Store** – Storing these vectors (with metadata) in a **Vector Database**, indexed for fast similarity search *(Day 9)*.

**End result:** a fully searchable **knowledge base**, sitting inside a vector store, ready to be queried.

```
raw data ──► loaders ──► chunks ──► embeddings ──► vector store  =  Knowledge Base ✅
```

---

## 7. Retrieval — Finding the Right Context

**Retrieval is the real-time process of finding the most relevant pieces of information from a pre-built index** (created during indexing) — based on the user's question.

It's like asking:

> *"From all the knowledge I have, which 3–5 chunks are most helpful to answer this query?"*

```
  www ─► Document ─► ▤ (raw doc)
         Loader          │
                          ▼
                    Text Splitter
                          │
              ┌─────┬─────┼─────┬─────┐
              ▼     ▼     ▼     ▼     ▼
             ▤     ▤     ▤     ▤     (chunks)
              │     │     │     │
              ▼     ▼     ▼     ▼
          ─────────────────────────
                 Embedding Model
          ─────────────────────────
              │     │     │     │
              ▼     ▼     ▼     ▼
            [▤▤▤] [▤▤▤] [▤▤▤] [▤▤▤]   ← vectors in the vector store
```

This entire diagram is essentially **Days 7, 8, and 9 combined into one pipeline** — document loading → text splitting → embedding → vector storage — which then gets searched by a **Retriever** *(Day 10)* to find those top 3–5 most relevant chunks for the user's query.

---

## 🔑 Quick Recap Table

| Concept | Key Idea |
|---|---|
| Parametric knowledge | LLM knowledge stored as parameters/weights, not as external facts |
| Hallucination | Confidently stating factually wrong information |
| 3 core problems | Private data, recent data, hallucination |
| Fine-tuning | Retrain the model on new/curated data — SFT, continued pretraining, RLHF |
| Fine-tuning drawbacks | Expensive, needs ML expertise, can't keep up with frequently changing data |
| In-context learning | Model learns from examples in the prompt, without updating weights |
| Emergent property | A capability that appears only once a model crosses a certain scale |
| RAG | Feed the model extra, relevant context at query time — no weight updates needed |
| 4 RAG steps | Indexing → Retrieval → Augmentation → Generation |
| Indexing sub-steps | Document ingestion → text chunking → embedding generation → vector store storage |



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

## Getting Started
    git clone https://github.com/SumitMARSS/LangChain_Models_Learning.git


## License
  This project is open for exploration and learning. Please credit the repository if you use its code or notes.
