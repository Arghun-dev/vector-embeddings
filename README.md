# vector-embeddings

Reference: https://supabase.com/blog/openai-embeddings-postgres-vector

---
- OpenAI Embedding Model -> text-embedding-3-small or text-embedding-3-large
- Supabase -> Open source postgresql database which has a vector database extension we can use
- Similarity search -> search using comparing embeddings in supabase `document_match` custom remote procedure function
- Langchain -> we need to create chunk of our long texts, `Langchain` is a framework that return chunks of a long text.
- Smarter Search
---

In this example based on user input we use similarity search to seach and find nearest match and generate a response using `ChatGPT` chat completion API.

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


## Building a simple search function

Finally, let's create an Edge Function to perform our similarity search:

```js
import express from 'express';
import cors from 'cors';
import { Configuration, OpenAIApi } from 'openai';
import { supabaseClient } from './lib/supabase';

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(
  cors({
    origin: '*',
    allowedHeaders: ['authorization', 'x-client-info', 'apikey', 'content-type'],
  })
);

// Preflight OPTIONS handler
app.options('*', (req, res) => res.sendStatus(200));

// POST route to handle the embedding and document matching
app.post('/', async (req, res) => {
  try {
    // Extract the search query from the request body
    const { query } = req.body;
    const input = query.replace(/\n/g, ' ');

    // Initialize OpenAI client
    const configuration = new Configuration({
      apiKey: '<YOUR_OPENAI_API_KEY>',
    });
    const openai = new OpenAIApi(configuration);

    // Create an embedding for the query
    const embeddingResponse = await openai.createEmbedding({
      model: 'text-embedding-ada-002',
      input,
    });
    const [{ embedding }] = embeddingResponse.data.data;

    // Call Supabase RPC to match documents using the embedding
    const { data: documents, error } = await supabaseClient.rpc('match_documents', {
      query_embedding: embedding,
      match_threshold: 0.78,
      match_count: 10,
    });

    if (error) throw error;

    res.json(documents);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message || 'Internal Server Error' });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
```

## Building a smarter search function

ChatGPT doesn't just return existing documents. It's able to assimilate a variety of information into a single, cohesive answer. To do this, we need to provide GPT with some relevant documents, and a prompt that it can use to formulate this answer.

One of the biggest challenges of OpenAI's text-davinci-003 completion model is the 4000 token limit. You must fit both your prompt and the resulting completion within the 4000 tokens. This makes it challenging if you wanted to prompt GPT-3 to answer questions about your own custom knowledge base that would never fit in a single prompt.

Embeddings can help solve this by splitting your prompts into a two-phased process:

Query your embedding database for the most relevant documents related to the question
Inject these documents as context for GPT-3 to reference in its answer
Here's another Edge Function that expands upon the simple example above:

```js
import express from 'express';
import cors from 'cors';
import { Configuration, OpenAIApi } from 'openai';
import GPT3Tokenizer from 'gpt3-tokenizer';
import { oneLine, stripIndent } from 'common-tags';
import { supabaseClient } from './lib/supabase';

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(
  cors({
    origin: '*',
    allowedHeaders: ['authorization', 'x-client-info', 'apikey', 'content-type'],
  })
);

// Preflight OPTIONS handler
app.options('*', (req, res) => res.sendStatus(200));

// POST route for handling the request
app.post('/', async (req, res) => {
  try {
    // Extract the query and normalize input
    const { query } = req.body;
    const input = query.replace(/\n/g, ' ');

    // Initialize OpenAI client
    const configuration = new Configuration({ apiKey: '<YOUR_OPENAI_API_KEY>' });
    const openai = new OpenAIApi(configuration);

    // Generate an embedding for the query
    const embeddingResponse = await openai.createEmbedding({
      model: 'text-embedding-ada-002',
      input,
    });
    const [{ embedding }] = embeddingResponse.data.data;

    // Fetch matching documents from Supabase
    const { data: documents, error } = await supabaseClient.rpc('match_documents', {
      query_embedding: embedding,
      match_threshold: 0.78,
      match_count: 10,
    });

    if (error) throw error;

    // Tokenize and concatenate context sections from documents
    const tokenizer = new GPT3Tokenizer({ type: 'gpt3' });
    let tokenCount = 0;
    let contextText = '';

    for (let i = 0; i < documents.length; i++) {
      const document = documents[i];
      const content = document.content;
      const encoded = tokenizer.encode(content);
      tokenCount += encoded.text.length;

      // Limit context to max 1500 tokens
      if (tokenCount > 1500) break;

      contextText += `${content.trim()}\n---\n`;
    }

    // Build the prompt using common-tags
    const prompt = stripIndent`${oneLine`
      You are a very enthusiastic Supabase representative who loves
      to help people! Given the following sections from the Supabase
      documentation, answer the question using only that information,
      outputted in markdown format. If you are unsure and the answer
      is not explicitly written in the documentation, say
      "Sorry, I don't know how to help with that."`}

      Context sections:
      ${contextText}

      Question: """
      ${query}
      """

      Answer as markdown (including related code snippets if available):
    `;

    // Generate the completion
    const completionResponse = await openai.createCompletion({
      model: 'text-davinci-003',
      prompt,
      max_tokens: 512,
      temperature: 0,
    });

    const {
      id,
      choices: [{ text }],
    } = completionResponse.data;

    res.json({ id, text });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message || 'Internal Server Error' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
```

## Streaming results

OpenAI API responses take longer to depending on the length of the “answer”. ChatGPT has a nice UX for this by streaming the response to the user immediately. You can see a similar effect for the Supabase docs:

The OpenAI API supports completion streaming with Server Side Events. Supabase Edge Functions are run Deno, which also supports Server Side Events. Check out this commit to see how we modified the Function above to build a streaming interface.

# Wrap up
Storing embeddings in Postgres opens a world of possibilities. You can combine your search function with telemetry functions, add an user-provided feedback (thumbs up/down), and make your search feel more integrated with your products.

The pgvector extension is available on all new Supabase projects today. To try it out, launch a new Postgres database: database.new
