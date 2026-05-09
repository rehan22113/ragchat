import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { readFileSync } from "fs";

// Step 1: Load + chunk (same as before)
const text = readFileSync("./sample.txt", "utf-8");
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 40 });
const chunks = await splitter.createDocuments([text]);

// Step 2: Create the embedding model
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });

// Step 3: Embed just the first chunk to see what a vector looks like
console.log("Embedding chunk 1...\n");
const vector = await embeddings.embedQuery(chunks[0].pageContent);

console.log("Chunk text:");
console.log(chunks[0].pageContent);
console.log(`\nVector length: ${vector.length} numbers`);
console.log(`First 10 numbers: [${vector.slice(0, 10).map(n => n.toFixed(4)).join(", ")}]`);

// Step 4: Embed two chunks and compare similarity
console.log("\n--- Similarity Test ---");

const textA = "What is LangChain used for?";
const textB = "LangChain is a framework for building LLM applications."; // similar meaning
const textC = "Ollama runs models locally on your machine.";             // different topic

const [vecA, vecB, vecC] = await Promise.all([
  embeddings.embedQuery(textA),
  embeddings.embedQuery(textB),
  embeddings.embedQuery(textC),
]);

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

const simAB = cosineSimilarity(vecA, vecB);
const simAC = cosineSimilarity(vecA, vecC);

console.log(`\nQuestion:  "${textA}"`);
console.log(`Similar:   "${textB}"`);
console.log(`Different: "${textC}"`);
console.log(`\nSimilarity score (A vs B): ${simAB.toFixed(4)}  ← should be HIGH`);
console.log(`Similarity score (A vs C): ${simAC.toFixed(4)}  ← should be LOW`);
console.log("\n1.0 = identical meaning, 0.0 = completely unrelated");
