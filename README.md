# LangChain_Models_Learning

## Overview

This repository documents my hands-on journey learning LangChain—a framework for building powerful language model applications. On the first day, I explored why LangChain is needed, the concept of semantic search, its main components, and the benefits of adopting this library.

## 📅 Day 1

### Why LangChain?

LangChain enables rapid development of applications using LLMs (Large Language Models), simplifying tasks such as text understanding, generation, and workflow automation. It abstracts away low-level details, allowing developers to focus on higher-level logic and integration.

### Key Benefits

- **Semantic Search:** LangChain empowers smarter search through meaning-based retrieval rather than keyword matching—improving accuracy in information retrieval.
- **Component Modularity:** Projects built with LangChain are easy to scale and extend due to its modular architecture.

### Components of LangChain

- **Models:** LLM providers (OpenAI, Hugging Face, etc.) for standard and custom text generation.
- **Prompts:** Templates and formatting for model inputs—configurable for different tasks and fine-tuning.
- **Chains:** Sequential pipelines linking models, prompts, and external tools to enable complex workflows.
- **Memory:** Mechanisms for passing context/history between chain steps and improving result relevance.
- **Indexes:** Data storage and retrieval systems designed for semantic search, chunking, and fast access.
- **Agents:** Autonomous decision-making units that dynamically select tools or chains based on task requirements.

## 📅 Day 2 

### 1. Prompt Engineering

- **Static Prompts**: Hardcoded text prompts that remain the same every time.
- **Dynamic Prompts**: Templates with variables that are filled at runtime.
- **Prompt Structures Tried**:
  - **Single Message**: A single user/system message sent to the LLM.
  - **List of Messages**: A sequence of system, user, and assistant messages for multi-turn context.
  - **Mixed Static + Dynamic Messages**: Combining static instructions with runtime user input dynamically.

#### 📝 Note: :-
    I had used Streamlit for some UI interaction while learning prompt engineering so that I can visualize things in an easy way.
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


## Getting Started
    git clone https://github.com/SumitMARSS/LangChain_Models_Learning.git


## License
  This project is open for exploration and learning. Please credit the repository if you use its code or notes.
