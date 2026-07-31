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