# Chat with PDF — Learning Roadmap

Build a "Chat with PDF" app while learning each concept from first principles.

---

## The Big Picture

```
PDF File
   ↓
[1. Chunking]        — split text into pieces
   ↓
[2. Embeddings]      — turn each chunk into a number vector
   ↓
[3. Vector DB]       — store & index those vectors
   ↓  ← User asks a question
[4. Retrieval]       — find the most relevant chunks
   ↓
[5. RAG]             — stuff chunks into a prompt + ask LLM
   ↓
Answer
```

LangChain is the glue framework that connects all of these steps.

---

## 1. Chunking

**What it is:** Splitting a large PDF into smaller text pieces so they fit inside a prompt.

**Why it matters:** LLMs have a limited context window. A 200-page PDF won't fit — but 20 relevant sentences will.

**Key concepts:**
- `chunk_size` — max characters per chunk (e.g. 1000)
- `chunk_overlap` — how many chars to repeat between chunks so context isn't lost at boundaries (e.g. 200)

**LangChain tools:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
```

**Strategies:**
| Strategy | Use when |
|---|---|
| Fixed-size (chars) | Simple, good default |
| Sentence splitter | Cleaner semantic boundaries |
| Semantic splitter | Groups by meaning (needs embeddings) |

---

## 2. Embeddings

**What it is:** A function that converts text → a list of numbers (a vector) that captures *meaning*.

**Why it matters:** Similar meaning → similar vectors. This lets you search by meaning, not just keywords.

**Example:**
```
"What is the refund policy?" → [0.12, -0.34, 0.87, ...]   (1536 numbers)
"How do I get a refund?"    → [0.11, -0.33, 0.85, ...]   (very close!)
"My cat is orange"          → [0.91,  0.22, -0.10, ...]  (far away)
```

**LangChain tools:**
```python
# OpenAI (requires API key, costs money)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Ollama (free, runs locally)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

**Mental model:** Think of vectors as GPS coordinates — similar texts land near each other in space.

---

## 3. Vector Database

**What it is:** A database designed to store vectors and find the nearest ones to a query vector fast.

**Why it matters:** Regular databases search by exact match. Vector DBs search by *similarity* — essential for semantic search.

**How similarity is measured:**
- **Cosine similarity** — angle between vectors (most common for text)
- **Euclidean distance** — straight-line distance

**Options:**
| DB | Type | Best for |
|---|---|---|
| FAISS | In-memory / local file | Learning, small projects |
| Chroma | Local, persistent | Local dev, medium projects |
| Pinecone | Cloud, managed | Production |
| Weaviate | Cloud/self-hosted | Production |

**LangChain tools:**
```python
# Chroma (easiest to start with)
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saves to disk
)
```

---

## 4. Retrieval

**What it is:** Given a user's question, find the top-K most relevant chunks from the vector DB.

**Why it matters:** You can't send the whole PDF to the LLM — only the parts that matter.

**How it works:**
1. Embed the user's question → query vector
2. Search vector DB for nearest chunk vectors
3. Return top-K chunks (e.g. top 4)

**LangChain tools:**
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Test it
docs = retriever.invoke("What is the refund policy?")
for doc in docs:
    print(doc.page_content)
```

**Retrieval strategies:**
| Strategy | What it does |
|---|---|
| Similarity search | Pure vector similarity (default) |
| MMR | Max Marginal Relevance — avoids duplicate chunks |
| Self-query | LLM rewrites the query before searching |
| Hybrid | Combines keyword + vector search |

---

## 5. RAG (Retrieval-Augmented Generation)

**What it is:** Retrieve relevant chunks → inject into LLM prompt → get a grounded answer.

**Why it matters:** LLMs hallucinate. RAG forces the LLM to answer *from the document*, not from training memory.

**The prompt pattern:**
```
Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{retrieved chunks}

Question: {user question}
Answer:
```

**LangChain tools:**
```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4o-mini")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is the refund policy?"})
print(result["result"])
```

---

## 6. OpenAI vs Ollama

| | OpenAI | Ollama |
|---|---|---|
| Cost | Pay per token | Free |
| Setup | API key only | Install Ollama locally |
| Speed | Fast (cloud) | Depends on your machine |
| Privacy | Data sent to OpenAI | 100% local |
| Models | GPT-4o, GPT-4o-mini | Llama 3, Mistral, Gemma... |

**OpenAI setup:**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

**Ollama setup:**
```bash
# Install: https://ollama.com
ollama pull llama3.2        # LLM
ollama pull nomic-embed-text  # embedding model
```
```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## 7. LangChain

**What it is:** A Python framework that chains together LLMs, retrievers, prompts, and tools.

**Core building blocks:**
| Block | What it does |
|---|---|
| `Document` | Text + metadata (source, page number) |
| `TextSplitter` | Chunk documents |
| `Embeddings` | Text → vector |
| `VectorStore` | Store + search vectors |
| `Retriever` | Interface to search a vector store |
| `LLM / ChatModel` | Generate text |
| `Chain` | Connect retriever + LLM into a pipeline |
| `Prompt Template` | Reusable prompt with variables |

**LCEL (LangChain Expression Language) — modern style:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
Answer based only on this context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is the refund policy?")
```

---

## Full App: Step-by-Step

### Step 1 — Install packages
```bash
pip install langchain langchain-openai langchain-ollama langchain-chroma \
            langchain-community pypdf chromadb python-dotenv
```

### Step 2 — Load PDF
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load()  # list of Document objects, one per page
```

### Step 3 — Chunk
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"{len(chunks)} chunks created")
```

### Step 4 — Embed + Store
```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
```

### Step 5 — Retrieve + Answer
```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not there, say "I don't know."

Context: {context}
Question: {question}
""")

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    q = input("Ask: ")
    if q == "exit": break
    print(chain.invoke(q))
```

---

## Learning Order (suggested)

- [ ] **Week 1** — Chunking + Embeddings: understand what vectors are, experiment with chunk sizes
- [ ] **Week 2** — Vector DB: store chunks in Chroma, run similarity searches manually
- [ ] **Week 3** — Retrieval: wire up a retriever, inspect what gets returned for different questions
- [ ] **Week 4** — RAG end-to-end: connect retriever to LLM, get your first answer from a PDF
- [ ] **Week 5** — Ollama vs OpenAI: swap models in and out, compare quality
- [ ] **Week 6** — Polish: add chat history, source citations, a Streamlit UI

---

## Quick Reference

```
PDF → load → chunk → embed → store in vector DB
                                    ↑
Question → embed → search vector DB → top K chunks → prompt + LLM → answer
```

| Term | One-liner |
|---|---|
| Chunk | A small piece of text from the PDF |
| Embedding | A list of numbers representing the meaning of text |
| Vector DB | A database that finds similar embeddings fast |
| Retriever | Searches the vector DB for relevant chunks |
| RAG | Retrieve context → augment prompt → generate answer |
| LangChain | Framework that chains all these steps together |
