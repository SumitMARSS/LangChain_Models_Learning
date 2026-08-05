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