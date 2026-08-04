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