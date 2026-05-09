import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { readFileSync } from "fs";

// Load the text file
const text = readFileSync("./sample.txt", "utf-8");

// Create the splitter
// chunk_size    = max characters per chunk
// chunk_overlap = how many chars repeat between chunks (so context isn't lost at edges)
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 40,
});

// Split into chunks
const chunks = await splitter.createDocuments([text]);

console.log(`Total chunks: ${chunks.length}\n`);

chunks.forEach((chunk, i) => {
  console.log(`--- Chunk ${i + 1} ---`);
  console.log(chunk.pageContent);
  console.log();
});
