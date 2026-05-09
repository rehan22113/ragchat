import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { readFileSync } from "fs";

// Step 1: Load + chunk (same as before)
const text = readFileSync("./sample.txt", "utf-8");
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 40 });
const chunks = await splitter.createDocuments([text]);
console.log(`Total chunks: ${chunks.length}`);

// Step 2: Embedding model
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });

// Step 3: Load existing vector store OR create a new one
import { existsSync } from "fs";

let vectorStore;
if (existsSync("./vector_store")) {
  console.log("Vector store already exists — loading it...\n");
  vectorStore = await HNSWLib.load("./vector_store", embeddings);
} else {
  console.log("No vector store found — embedding all chunks and saving...\n");
  vectorStore = await HNSWLib.fromDocuments(chunks, embeddings);
  await vectorStore.save("./vector_store");
  console.log("Saved to ./vector_store\n");
}

// Step 4: Search — ask a question, get relevant chunks back
const question = "What is RAG?";
console.log(`Question: "${question}"\n`);

const results = await vectorStore.similaritySearchWithScore(question, 3);

results.forEach(([doc, score], i) => {
  console.log(`--- Result ${i + 1} (score: ${score.toFixed(4)}) ---`);
  console.log(doc.pageContent);
  console.log();
});
