import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import * as readline from "readline";

// Load vector store
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
const vectorStore = await HNSWLib.load("./vector_store", embeddings);
const retriever = vectorStore.asRetriever({ k: 2 });

// LLM
const llm = new ChatOllama({ model: "deepseek-r1:8b" });

// Prompt — tells the LLM to answer ONLY from the context
const prompt = ChatPromptTemplate.fromTemplate(`
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:
`);

// Chain: question → retrieve chunks → format → prompt → LLM → answer
const chain = RunnableSequence.from([
  {
    context: async (input) => {
      const docs = await retriever.invoke(input.question);
      return docs.map(d => d.pageContent).join("\n\n");
    },
    question: (input) => input.question,
  },
  prompt,
  llm,
  new StringOutputParser(),
]);

// Chat loop — keep asking questions until you type "exit"
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise(resolve => rl.question(q, resolve));

console.log("Chat with your document! Type 'exit' to quit.\n");

while (true) {
  const question = await ask("You: ");
  if (question.toLowerCase() === "exit") { rl.close(); break; }

  process.stdout.write("AI: ");
  const stream = await chain.stream({ question });
  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
  console.log("\n");
}
