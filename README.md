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



## Getting Started
    git clone https://github.com/SumitMARSS/LangChain_Models_Learning.git


## License
  This project is open for exploration and learning. Please credit the repository if you use its code or notes.
