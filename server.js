import "dotenv/config";
import express from "express";
import multer from "multer";
import fs from "fs";
import crypto from "crypto";
import pg from "pg";
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

// Postgres connection
const db = new pg.Pool({ connectionString: process.env.DATABASE_URL });

// In-memory vector stores (loaded from disk on startup)
const vectorStores = new Map();

// ── Load all sessions from DB + vector stores from disk on startup ──
async function loadSessions() {
  const { rows } = await db.query("SELECT * FROM sessions ORDER BY created_at ASC");
  for (const row of rows) {
    const storePath = `./vector_stores/${row.id}`;
    if (fs.existsSync(storePath)) {
      try {
        const vs = await HNSWLib.load(storePath, embeddings);
        vectorStores.set(row.id, vs);
        console.log(`Loaded session: ${row.file_name}`);
      } catch (e) {
        console.warn(`Could not load vector store for ${row.id}`);
      }
    }
  }
}

// ── Upload + Index PDF ──────────────────────────────────────────
app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    const loader = new PDFLoader(req.file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
    const chunks = await splitter.splitDocuments(docs);

    const vectorStore = await HNSWLib.fromDocuments(chunks, embeddings);

    const sampleText = chunks.slice(0, 5).map(c => c.pageContent).join("\n");
    const summaryResult = await llm.invoke(
      `In 2 sentences, what topics does this document cover? Be specific.\n\n${sampleText}`
    );
    const summary = summaryResult.content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    const sessionId = crypto.randomUUID();

    // Save vector store to disk
    const storePath = `./vector_stores/${sessionId}`;
    fs.mkdirSync(storePath, { recursive: true });
    await vectorStore.save(storePath);
    vectorStores.set(sessionId, vectorStore);

    // Save session to Postgres
    await db.query(
      `INSERT INTO sessions (id, file_name, pages, chunks, summary) VALUES ($1, $2, $3, $4, $5)`,
      [sessionId, req.file.originalname, docs.length, chunks.length, summary]
    );

    res.json({ sessionId, pages: docs.length, chunks: chunks.length, summary });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Get all sessions ────────────────────────────────────────────
app.get("/sessions", async (req, res) => {
  const { rows } = await db.query("SELECT * FROM sessions ORDER BY created_at ASC");
  res.json(rows);
});

// ── Get messages for a session ──────────────────────────────────
app.get("/sessions/:id/messages", async (req, res) => {
  const { rows } = await db.query(
    "SELECT role, content FROM messages WHERE session_id = $1 ORDER BY created_at ASC",
    [req.params.id]
  );
  res.json(rows);
});

// ── Delete session ──────────────────────────────────────────────
app.delete("/sessions/:id", async (req, res) => {
  const { id } = req.params;
  await db.query("DELETE FROM sessions WHERE id = $1", [id]);
  vectorStores.delete(id);
  const storePath = `./vector_stores/${id}`;
  if (fs.existsSync(storePath)) fs.rmSync(storePath, { recursive: true });
  res.json({ success: true });
});

// ── Chat ────────────────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const { sessionId, question } = req.body;

  const session = await db.query("SELECT * FROM sessions WHERE id = $1", [sessionId]);
  if (!session.rows.length) return res.status(404).json({ error: "Session not found." });

  const vectorStore = vectorStores.get(sessionId);
  if (!vectorStore) return res.status(404).json({ error: "Vector store not loaded." });

  try {
    // Load last 8 messages from DB for history
    const { rows: history } = await db.query(
      "SELECT role, content FROM messages WHERE session_id = $1 ORDER BY created_at DESC LIMIT 8",
      [sessionId]
    );
    const historyText = history.reverse().map(m =>
      `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`
    ).join("\n");

    const retriever = vectorStore.asRetriever({ k: 3 });
    const { summary } = session.rows[0];

    const prompt = ChatPromptTemplate.fromTemplate(`
You are a friendly and helpful AI assistant for a PDF document.

RULES:
1. If the question is a greeting or small talk — respond naturally and warmly.
2. If the question relates to the document — answer using ONLY the context below.
3. If the question is unrelated — mention what the document covers and invite them to ask about it.
4. Use conversation history for follow-up context.

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
        summary: () => summary,
        history: () => historyText || "No previous conversation.",
      },
      prompt,
      llm,
      new StringOutputParser(),
    ]);

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");

    let fullText = "";
    const stream = await chain.stream({ question });
    for await (const chunk of stream) {
      fullText += chunk;
      res.write(`data: ${JSON.stringify({ token: chunk })}\n\n`);
    }
    res.write("data: [DONE]\n\n");
    res.end();

    // Save both messages to DB after response is complete
    const cleanAnswer = fullText.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    await db.query("INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3)", [sessionId, "user", question]);
    await db.query("INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3)", [sessionId, "ai", cleanAnswer]);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

await loadSessions();
app.listen(3000, () => console.log("Server running at http://localhost:3000"));
