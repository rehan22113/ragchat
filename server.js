import express from "express";
import multer from "multer";
import fs from "fs";
import crypto from "crypto";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const app = express();
app.use(express.json());
app.use(express.static("public"));

const upload = multer({ dest: "uploads/" });
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
const llm = new ChatOllama({ model: "deepseek-r1:8b" });

// sessionId → { vectorStore, docSummary, fileName, pages, chunks, createdAt }
const sessions = new Map();

// ── Upload + Index PDF ──────────────────────────────────────────
app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    const loader = new PDFLoader(req.file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
    const chunks = await splitter.splitDocuments(docs);

    const vectorStore = await HNSWLib.fromDocuments(chunks, embeddings);

    // Auto-summarize what the doc covers
    const sampleText = chunks.slice(0, 5).map(c => c.pageContent).join("\n");
    const summaryResult = await llm.invoke(
      `In 2 sentences, what topics does this document cover? Be specific.\n\n${sampleText}`
    );
    const docSummary = summaryResult.content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    const sessionId = crypto.randomUUID();
    sessions.set(sessionId, {
      vectorStore,
      docSummary,
      fileName: req.file.originalname,
      pages: docs.length,
      chunks: chunks.length,
      createdAt: new Date().toISOString(),
    });

    res.json({ sessionId, fileName: req.file.originalname, pages: docs.length, chunks: chunks.length, summary: docSummary });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── List Sessions ───────────────────────────────────────────────
app.get("/sessions", (req, res) => {
  const list = Array.from(sessions.entries()).map(([id, s]) => ({
    id,
    fileName: s.fileName,
    pages: s.pages,
    chunks: s.chunks,
    summary: s.docSummary,
    createdAt: s.createdAt,
  }));
  res.json(list);
});

// ── Delete Session ──────────────────────────────────────────────
app.delete("/sessions/:id", (req, res) => {
  sessions.delete(req.params.id);
  res.json({ success: true });
});

// ── Chat ────────────────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const { sessionId, question, history = [] } = req.body;
  const session = sessions.get(sessionId);
  if (!session) return res.status(404).json({ error: "Session not found." });

  try {
    const retriever = session.vectorStore.asRetriever({ k: 3 });

    // Build conversation history string for context
    const historyText = history.slice(-6).map(m =>
      `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`
    ).join("\n");

    const prompt = ChatPromptTemplate.fromTemplate(`
You are a friendly and helpful AI assistant for a PDF document.

RULES:
1. If the question is a greeting or small talk — respond naturally and warmly.
2. If the question relates to the document — answer using ONLY the context below.
3. If the question is unrelated to the document — mention what the document covers and invite them to ask about it.
4. Use the conversation history for context when answering follow-up questions.

Document covers: {summary}

Conversation history:
{history}

Context from document:
{context}

Question: {question}

Answer:`);

    const chain = RunnableSequence.from([
      {
        context: async (input) => {
          const docs = await retriever.invoke(input.question);
          return docs.map(d => d.pageContent).join("\n\n");
        },
        question: (input) => input.question,
        summary: () => session.docSummary,
        history: () => historyText || "No previous conversation.",
      },
      prompt,
      llm,
      new StringOutputParser(),
    ]);

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");

    const stream = await chain.stream({ question });
    for await (const chunk of stream) {
      res.write(`data: ${JSON.stringify({ token: chunk })}\n\n`);
    }
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log("Server running at http://localhost:3000"));
