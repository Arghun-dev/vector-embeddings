# vector-embeddings

Reference: https://supabase.com/blog/openai-embeddings-postgres-vector

in this code I'm using `Semantic Search` using `Vector embeddings and Database` and create a conversational response using OpenAI gpt-4 model.

```js
import { openai, supabase } from './config.js';

// User query about podcasts
const query = "An episode Elon Musk would enjoy";
main(query);

// Bring all function calls together
async function main(input) {
  const embedding = await createEmbedding(input);
  const match = await findNearestMatch(embedding);
  await getChatCompletion(match, input);
}

// Create an embedding vector representing the input text
async function createEmbedding(input) {
  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input
  });
  return embeddingResponse.data[0].embedding;
}

// Query Supabase and return a semantically matching text chunk
async function findNearestMatch(embedding) {
  const { data } = await supabase.rpc('match_documents', {
    query_embedding: embedding,
    match_threshold: 0.50,
    match_count: 1
  });
  return data[0].content;
}

// Use OpenAI to make the response conversational
const chatMessages = [{
    role: 'system',
    content: `You are an enthusiastic podcast expert who loves recommending podcasts to people. You will be given two pieces of information - some context about podcasts episodes and a question. Your main job is to formulate a short answer to the question using the provided context. If you are unsure and cannot find the answer in the context, say, "Sorry, I don't know the answer." Please do not make up the answer.` 
}];

async function getChatCompletion(text, query) {
  chatMessages.push({
    role: 'user',
    content: `Context: ${text} Question: ${query}`
  });
  
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: chatMessages,
    temperature: 0.5,
    frequency_penalty: 0.5
  });

  console.log(response.choices[0].message.content);
}
```

## Long Text File

If you have a long text file it's better you first chunk your text file.


**Effective chunking**

1. Ensure content is free from unnecessary or irrelevant information
2. Remove HTML tags, characters, or symbols that can affect your embeddings
3. Correct typos
4. Remove any repeated text, and standardize your text formatting


It's common to use a framework to manage the chunking -> `Langchain` is one of the most popular frameworks for developing AI powered apps, and it includes incredible tools for splitting text.

```js
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// LangChain text splitter
async function splitDocument() {
  const response = await fetch('podcasts.txt');
  const text = await response.text();
  const splitter = new RecursiveCharacterTextSplitter({
   chunkSize: 150,
   chunkOverlap: 15
  })

  const output = await splitter.createDocuments([text])
}
```

### Choosing a chunk size:

. Depend on the type of content: short content vs. large documents
. Consider the embedding model and its token limits
. User queries: short and specific vs. longer and more detailed
. Consider how you'll use the retrieved results


**RecursiveCharacterTextSplitter:** splits the text iteratively into optimal chunk sizes, if the initial split doesn't produce the desired size or structure, it repeatedly or recursively calls itself using a different separators that might produce better results, and it does that until it reaches the desired size. Because of this efficiency, `Recuresive` splitting is often recommended for generic text.

- Shoter chunks capture precise meaning but might miss wider context.
- Longer chunks grasp more context but can produce too broad a scope of information, potentially leading to confusion for the model processing it. So, you'll likely have to experiment with various chunk sizes.
‍‍‍‍‍‍

### Generate chunk vector embeddings and store it to supabase Vector Database

```js
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { openai, supabase } from './config.js';

/** 
 * Split movies.txt into text chunks.
 * Returns LangChain's array of Document objects.
 */
async function splitDocument(documentPath) {
  // Fetch the text file and wait for its content
  const response = await fetch(documentPath);
  const text = await response.text();
  
  // Initialize the splitter with desired chunk size and overlap
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 150,
    chunkOverlap: 15,
  });
  
  // Create documents (each document is an object with a pageContent property)
  const output = await splitter.createDocuments([text]);
  return output;
}

/**
 * Create an embedding for each text chunk and store both the text and its embedding in Supabase.
 */
async function createAndStoreEmbeddings() {
  try {
    const chunkData = await splitDocument("movies.txt");
    
    for (const cd of chunkData) {
      // Generate embedding for the chunk's text
      const embeddingResponse = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: cd.pageContent,
        encoding_format: "float",
      });
      
      // Extract the embedding vector from the response
      const embeddedChunk = embeddingResponse.data[0].embedding;
      
      // Insert the text chunk and its embedding into the Supabase 'embeddings' table
      const { data, error } = await supabase
        .from('embeddings')
        .insert([
          { text: cd.pageContent, embedding: embeddedChunk }
        ]);
      
      if (error) {
        console.error("Error inserting embedding:", error);
      } else {
        console.log("Inserted embedding for chunk (first 30 chars):", cd.pageContent.slice(0, 30));
      }
    }
  } catch (err) {
    console.error("Error in createAndStoreEmbeddings:", err);
  }
}

// Execute the function
createAndStoreEmbeddings();
```
