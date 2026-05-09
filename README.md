# RAGChat — Chat with any PDF using AI

A fully local "Chat with PDF" app built with **LangChain.js**, **Ollama**, and **DeepSeek R1**. No OpenAI API key needed — everything runs on your machine.

![RAGChat Demo](https://img.shields.io/badge/AI-DeepSeek_R1-blue) ![LangChain](https://img.shields.io/badge/LangChain-JS-green) ![Ollama](https://img.shields.io/badge/Ollama-local-orange)

## How it works

```
PDF → Chunks → Embeddings → Vector DB
                                 ↑
Question → Embed → Search → Top chunks → DeepSeek R1 → Answer
```

1. **Upload a PDF** — the app splits it into chunks and embeds each one using `nomic-embed-text`
2. **Ask a question** — your question is embedded and matched against the stored chunks
3. **Get an answer** — the most relevant chunks are passed to DeepSeek R1 which answers from the document

## Tech Stack

| Layer | Tool |
|---|---|
| Framework | LangChain.js |
| LLM | DeepSeek R1 8B (via Ollama) |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector DB | HNSWLib (local, no server needed) |
| Backend | Express.js |
| Frontend | Vanilla HTML/CSS/JS |

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org) v18+
- [Ollama](https://ollama.com) installed and running

### 1. Pull required models
```bash
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

### 2. Clone and install
```bash
git clone https://github.com/rehan22113/ragchat.git
cd ragchat
npm install
```

### 3. Run
```bash
ollama serve       # in one terminal tab
node server.js     # in another tab
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features

- Drag & drop PDF upload
- Streaming responses (word by word)
- Thinking indicator while DeepSeek R1 reasons
- Answers casual questions (greetings, small talk)
- Tells you what the document covers when asked off-topic questions
- 100% local — no data leaves your machine
