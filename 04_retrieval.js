import { OllamaEmbeddings } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

// Load the already saved vector store (no re-embedding needed)
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
const vectorStore = await HNSWLib.load("./vector_store", embeddings);

// Create a retriever — this is the official LangChain way to search
// k: how many chunks to return
const retriever = vectorStore.asRetriever({ k: 2 });

// Ask different questions and see which chunks come back
const questions = [
  "What is chunking?",
  "How does Ollama work?",
  "What is a vector database?",
];

for (const question of questions) {
  console.log(`\nQuestion: "${question}"`);
  console.log("─".repeat(50));

  const docs = await retriever.invoke(question);

  docs.forEach((doc, i) => {
    console.log(`\nChunk ${i + 1}:`);
    console.log(doc.pageContent);
  });
}
