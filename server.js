import express from "express";
import multer from "multer";
import fs from "fs";
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

let vectorStore = null;
let docSummary = "";   // short summary of what the PDF covers

// ── Upload + Index PDF ──────────────────────────────────────────
app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    const loader = new PDFLoader(req.file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
    const chunks = await splitter.splitDocuments(docs);

    if (fs.existsSync("./vector_store")) fs.rmSync("./vector_store", { recursive: true });
    vectorStore = await HNSWLib.fromDocuments(chunks, embeddings);
    await vectorStore.save("./vector_store");

    // Build a short summary of the document so the AI can tell users what it knows
    const sampleText = chunks.slice(0, 5).map(c => c.pageContent).join("\n");
    const summaryResult = await llm.invoke(
      `In 2 sentences, what topics does this document cover? Be specific.\n\n${sampleText}`
    );
    docSummary = summaryResult.content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    res.json({ success: true, chunks: chunks.length, pages: docs.length, summary: docSummary });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Chat ────────────────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const { question } = req.body;
  if (!vectorStore) return res.status(400).json({ error: "No PDF uploaded yet." });

  try {
    const retriever = vectorStore.asRetriever({ k: 3 });

    const prompt = ChatPromptTemplate.fromTemplate(`
You are a friendly and helpful AI assistant for a PDF document.

RULES:
1. If the question is a greeting or small talk (hi, hello, how are you, thanks, etc.) — respond naturally and warmly. You can mention you are here to help with the document.
2. If the question is related to the document context — answer it clearly and accurately using ONLY the context.
3. If the question is NOT related to the document context — do NOT say "I don't know". Instead say what you DO know: mention the document topics and invite the user to ask about those.

Document covers: {summary}

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
        summary: () => docSummary,
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
