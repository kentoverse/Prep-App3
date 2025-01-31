# The 5 ways of building a RAG Application
## Prep-App3


## E2E Integration Architecture for RAG Project

### Key Takeaways for the Interview

1. **Agile Leadership** - Led Agile-based iterative development with a focus on cross-team collaboration.
2. **Scalable System Architecture** - Implemented secure, cloud-based CI/CD pipelines for seamless deployments.
3. **Trust in AI** - Addressed AI skepticism by demonstrating how RAG ensures content integrity.
4. **Hands-on Development** - Built AI chat engine and RAG-based content retrieval using Python, FastAPI, and Next.js.
5. **Long-Term Vision** - Designed a roadmap for AI expansion, ensuring future product growth and sustainability.

## Integration Architecture Table

| **Integration System**      | **AWS Product**                         | **GCP Product**                              | **Role of Developer**                                       |
|----------------------------|---------------------------------|---------------------------------|------------------------------------------------|
| **Event-Driven Messaging** | Amazon MSK (Kafka), SQS, SNS   | Pub/Sub, Cloud Tasks            | Configure event-driven pipelines, optimize message queues, ensure scalability |
| **AI Integration (RAG)**   | Amazon Bedrock, OpenSearch     | Vertex AI, ElasticSearch, Pinecone | Develop RAG pipeline, integrate vector search, fine-tune retrieval |
| **Frontend Development**   | Amplify, AppSync (GraphQL)     | Firebase, Cloud Functions       | Build UI, integrate GraphQL APIs, optimize AI-driven UX |
| **Workflow Automation**    | Step Functions, AWS Lambda     | Workflows, Cloud Run            | Define AI-driven workflows, automate approvals, integrate AI decisions |
| **Cloud-Native Deployment** | EKS (Kubernetes), Lambda, IAM  | GKE, Cloud Run, IAM              | Deploy microservices, ensure scalability, optimize CI/CD |

## Challenges & Solutions

| **Challenge**                                | **Solution**                             |
|---------------------------------------------|----------------------------------------|
| **Scalability for AI queries**              | Use **autoscaling** with Kubernetes (EKS/GKE) and serverless functions (AWS Lambda, Cloud Run). |
| **High latency in vector search**           | Optimize embeddings retrieval with **Pinecone/ElasticSearch caching**. |
| **Security concerns with AI-generated data** | Implement **OAuth2 (Azure AD)**, role-based access (RBAC), and encryption (AES-256, TLS). |
| **Complex workflow automation**             | Leverage **Camunda BPM + AI decision-making** for intelligent approvals. |
| **Integration of AI with existing tools**   | Utilize **GraphQL APIs** for seamless frontend-backend connectivity. |

## Final Thoughts

This architecture ensures a **scalable, event-driven, AI-powered system** with cloud-native deployments on AWS & GCP, integrating **Kafka, AI RAG, Camunda BPM, and Kubernetes** to power an **automated, intelligent workflow.** 🚀




# **Retrieval-Augmented Generation (RAG) System**

## **Overview**
This guide explains how to tokenize words, embed them into numerical vectors, and integrate them into a **Retrieval-Augmented Generation (RAG) system**. The process follows a **five-step parallel implementation** that aligns with five user data flows.

---

## **RAG System - Five Implementations & Five Data Flows**

### **1️⃣ Tokenization (User Input Processing)**
Tokenization is the first step, breaking text into words, subwords, or characters.

#### **1.1 Using Hugging Face Tokenizers**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "How do I tokenize and embed words for RAG?"
tokens = tokenizer(text, return_tensors="pt")

print(tokens)
```
✅ **Result:** Tokenized input IDs and attention masks for the model.

#### **1.2 Using NLTK (Basic Word Tokenization)**
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "How do I tokenize and embed words for RAG?"
tokens = word_tokenize(text)

print(tokens)
```
✅ **Result:** `['How', 'do', 'I', 'tokenize', 'and', 'embed', 'words', 'for', 'RAG', '?']`

---

### **2️⃣ Embedding Tokens (Numerical Vector Representation)**
Tokenized words are converted into numerical vectors for retrieval and processing.

#### **2.1 Using OpenAI Embeddings**
```python
import openai

openai.api_key = "your-api-key"

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="How do I tokenize and embed words for RAG?"
)

embedding_vector = response["data"][0]["embedding"]
print(embedding_vector[:5])  # Printing first 5 values for preview
```
✅ **Result:** High-dimensional vector representation of text.

#### **2.2 Using Sentence Transformers (Local Embeddings)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_vector = model.encode(["How do I tokenize and embed words for RAG?"])

print(embedding_vector.shape)
```
✅ **Result:** A `768-dimensional` vector representation.

---

### **3️⃣ Storing and Retrieving Embeddings (Vector Database Management)**
Once embeddings are generated, they must be stored and retrieved efficiently.

#### **3.1 Using FAISS (Facebook AI Similarity Search)**
```python
import faiss
import numpy as np

dimension = 384  # Embedding size (depends on model)
index = faiss.IndexFlatL2(dimension)

# Sample embeddings (normally generated from a model)
embedding_vectors = np.random.random((10, dimension)).astype("float32")
index.add(embedding_vectors)  # Adding embeddings to FAISS

query_vector = np.random.random((1, dimension)).astype("float32")
_, nearest_neighbors = index.search(query_vector, k=3)

print("Nearest neighbors:", nearest_neighbors)
```
✅ **Result:** Retrieves the most similar vectors based on user query.

---

### **4️⃣ Integrating with a RAG Pipeline (Data Flow & Query Handling)**

The **RAG pipeline** follows these five steps:
1. **User Query → Tokenization → Embedding Generation**
2. **Search in a Vector Database (FAISS, Weaviate, Pinecone, Chroma)**
3. **Retrieve Relevant Context**
4. **Pass Context + Query to LLM for Answer Generation**
5. **Return Response to the User**

#### **4.1 Example Using LlamaIndex**
```python
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

# Load documents
documents = SimpleDirectoryReader("your_data_folder").load_data()

# Index documents
index = GPTVectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is tokenization in NLP?")

print(response)
```
✅ **Result:** Retrieves relevant text and generates an answer.

---

### **5️⃣ LangChain-Based RAG Implementation (End-to-End System)**
LangChain enables an end-to-end RAG pipeline with query processing, vector search, and LLM response generation.

#### **5.1 LangChain with FAISS and OpenAI**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Load FAISS index
vectorstore = FAISS.load_local("faiss_index", embedding_model)

# Set up Retrieval-based QA
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# Query the RAG system
query = "What is tokenization in NLP?"
response = qa.run(query)

print("Response:", response)
```
✅ **Result:** Full RAG pipeline using LangChain and FAISS for retrieval.

---

## **Conclusion**
| Step | Implementation | Technology |
|------|-----------------|--------------|
| 1️⃣ **Tokenization** | Convert text into tokens | Hugging Face, NLTK |
| 2️⃣ **Embedding** | Convert tokens to vectors | OpenAI, Sentence-Transformers |
| 3️⃣ **Storage & Retrieval** | Store embeddings for similarity search | FAISS, Weaviate, Pinecone |
| 4️⃣ **RAG Pipeline** | Retrieve and generate responses | LlamaIndex, LangChain |
| 5️⃣ **Full RAG System** | Query handling, vector search, LLM response | LangChain, OpenAI |

---

## **Next Steps**
- **Expand RAG with Pinecone or Weaviate for large-scale retrieval**
- **Integrate with a chatbot UI using FastAPI or Streamlit**
- **Optimize embeddings for domain-specific knowledge**

🚀 **Would you like a frontend UI (React/Next.js) for this RAG system?**


# **Retrieval-Augmented Generation (RAG) System**

## **Overview**
This guide explains how to tokenize words, embed them into numerical vectors, and integrate them into a **Retrieval-Augmented Generation (RAG) system**. The process follows a **five-phase parallel implementation** that aligns with five user data flows.

---

## **Phase 1: Tokenization (User Input Processing)**
Tokenization is the first step, breaking text into words, subwords, or characters.

### **Implementation Steps**
#### **1.1 Using Hugging Face Tokenizers**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "How do I tokenize and embed words for RAG?"
tokens = tokenizer(text, return_tensors="pt")

print(tokens)
```
✅ **Result:** Tokenized input IDs and attention masks for the model.

#### **1.2 Using NLTK (Basic Word Tokenization)**
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "How do I tokenize and embed words for RAG?"
tokens = word_tokenize(text)

print(tokens)
```
✅ **Result:** `['How', 'do', 'I', 'tokenize', 'and', 'embed', 'words', 'for', 'RAG', '?']`

---

## **Phase 2: Embedding Tokens (Numerical Vector Representation)**
Tokenized words are converted into numerical vectors for retrieval and processing.

### **Implementation Steps**
#### **2.1 Using OpenAI Embeddings**
```python
import openai

openai.api_key = "your-api-key"

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="How do I tokenize and embed words for RAG?"
)

embedding_vector = response["data"][0]["embedding"]
print(embedding_vector[:5])  # Printing first 5 values for preview
```
✅ **Result:** High-dimensional vector representation of text.

#### **2.2 Using Sentence Transformers (Local Embeddings)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_vector = model.encode(["How do I tokenize and embed words for RAG?"])

print(embedding_vector.shape)
```
✅ **Result:** A `768-dimensional` vector representation.

---

## **Phase 3: Storing and Retrieving Embeddings (Vector Database Management)**
Once embeddings are generated, they must be stored and retrieved efficiently.

### **Implementation Steps**
#### **3.1 Using FAISS (Facebook AI Similarity Search)**
```python
import faiss
import numpy as np

dimension = 384  # Embedding size (depends on model)
index = faiss.IndexFlatL2(dimension)

# Sample embeddings (normally generated from a model)
embedding_vectors = np.random.random((10, dimension)).astype("float32")
index.add(embedding_vectors)  # Adding embeddings to FAISS

query_vector = np.random.random((1, dimension)).astype("float32")
_, nearest_neighbors = index.search(query_vector, k=3)

print("Nearest neighbors:", nearest_neighbors)
```
✅ **Result:** Retrieves the most similar vectors based on user query.

---

## **Phase 4: Integrating with a RAG Pipeline (Data Flow & Query Handling)**

The **RAG pipeline** follows these five steps:
1. **User Query → Tokenization → Embedding Generation**
2. **Search in a Vector Database (FAISS, Weaviate, Pinecone, Chroma)**
3. **Retrieve Relevant Context**
4. **Pass Context + Query to LLM for Answer Generation**
5. **Return Response to the User**

### **Implementation Steps**
#### **4.1 Example Using LlamaIndex**
```python
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

# Load documents
documents = SimpleDirectoryReader("your_data_folder").load_data()

# Index documents
index = GPTVectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is tokenization in NLP?")

print(response)
```
✅ **Result:** Retrieves relevant text and generates an answer.

---

## **Phase 5: LangChain-Based RAG Implementation (End-to-End System)**
LangChain enables an end-to-end RAG pipeline with query processing, vector search, and LLM response generation.

### **Implementation Steps**
#### **5.1 LangChain with FAISS and OpenAI**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Load FAISS index
vectorstore = FAISS.load_local("faiss_index", embedding_model)

# Set up Retrieval-based QA
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# Query the RAG system
query = "What is tokenization in NLP?"
response = qa.run(query)

print("Response:", response)
```
✅ **Result:** Full RAG pipeline using LangChain and FAISS for retrieval.

---

## **Conclusion**
| Phase | Implementation | Technology |
|------|-----------------|--------------|
| 1️⃣ **Tokenization** | Convert text into tokens | Hugging Face, NLTK |
| 2️⃣ **Embedding** | Convert tokens to vectors | OpenAI, Sentence-Transformers |
| 3️⃣ **Storage & Retrieval** | Store embeddings for similarity search | FAISS, Weaviate, Pinecone |
| 4️⃣ **RAG Pipeline** | Retrieve and generate responses | LlamaIndex, LangChain |
| 5️⃣ **Full RAG System** | Query handling, vector search, LLM response | LangChain, OpenAI |

---

## **Next Steps**
- **Expand RAG with Pinecone or Weaviate for large-scale retrieval**
- **Integrate with a chatbot UI using FastAPI or Streamlit**
- **Optimize embeddings for domain-specific knowledge**

🚀 **Would you like a frontend UI (React/Next.js) for this RAG system?**

📌 Full-Stack RAG System: Spring Boot (Backend) + React (Frontend)

This example will cover:
	1.	Spring Boot Backend:
	•	Handles RAG pipeline.
	•	Accepts queries, retrieves relevant documents, and generates text/image output.
	2.	React Frontend:
	•	Handles user input and displays text or image responses dynamically.

📌 Step 1: Spring Boot Backend

The backend will process user queries, generate responses using OpenAI, and return either text or an image URL.

1️⃣ Modify RagService.java

This service:
	•	Uses OpenAI for text and DALL·E for image generation.
	•	Determines response type based on user input.

package com.example.rag;

import org.springframework.stereotype.Service;
import com.theokanning.openai.service.OpenAiService;
import com.theokanning.openai.completion.CompletionRequest;
import com.theokanning.openai.image.CreateImageRequest;
import java.util.HashMap;
import java.util.Map;

@Service
public class RagService {

    private final OpenAiService openAiService = new OpenAiService("your-openai-api-key");

    public Map<String, String> processQuery(String query) {
        Map<String, String> response = new HashMap<>();

        if (query.toLowerCase().contains("image")) {
            // Generate an AI image using DALL·E
            CreateImageRequest imageRequest = CreateImageRequest.builder()
                    .prompt(query.replace("image", "").trim())
                    .size("1024x1024")
                    .build();

            String imageUrl = openAiService.createImage(imageRequest).getData().get(0).getUrl();
            response.put("type", "image");
            response.put("content", imageUrl);

        } else {
            // Generate text response
            CompletionRequest completionRequest = CompletionRequest.builder()
                    .model("gpt-4")
                    .prompt(query)
                    .maxTokens(100)
                    .build();

            String textResponse = openAiService.createCompletion(completionRequest).getChoices().get(0).getText();
            response.put("type", "text");
            response.put("content", textResponse);
        }

        return response;
    }
}

✅ Automatically detects if user wants text or an image and returns appropriate output.

2️⃣ Modify RagController.java

This REST endpoint will return either text or an image URL based on the query.

package com.example.rag;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.Map;

@RestController
@RequestMapping("/api/rag")
public class RagController {

    @Autowired
    private RagService ragService;

    @PostMapping("/query")
    public ResponseEntity<Map<String, String>> queryRagSystem(@RequestBody QueryRequest request) {
        Map<String, String> response = ragService.processQuery(request.getQuery());
        return ResponseEntity.ok(response);
    }
}

class QueryRequest {
    private String query;
    public String getQuery() { return query; }
}

✅ Returns JSON response with type (text or image) and content.

📌 Step 2: React Frontend

The frontend will:
	•	Handle user input.
	•	Display either text or an image based on the API response.

1️⃣ Install Dependencies

Run:

npx create-next-app rag-frontend
cd rag-frontend
npm install axios

2️⃣ Create API Call in api.js

Helper function to fetch responses from the Spring Boot backend.

import axios from 'axios';

const API_URL = "http://localhost:8080/api/rag/query";

export const queryRagSystem = async (query) => {
    const response = await axios.post(API_URL, { query });
    return response.data; // { type: "text" | "image", content: "..."}
};

3️⃣ Create UI Component (pages/index.js)

Handles:
	•	State management for input and response.
	•	Conditional rendering for text vs image responses.

import { useState } from "react";
import { queryRagSystem } from "../api";

export default function Home() {
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const result = await queryRagSystem(query);
        setResponse(result);
    };

    return (
        <div className="container">
            <h1>RAG System</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your query (Type 'image of a cat')"
                />
                <button type="submit">Submit</button>
            </form>

            {response && (
                <div>
                    {response.type === "text" ? (
                        <p><strong>Response:</strong> {response.content}</p>
                    ) : (
                        <img src={response.content} alt="Generated" style={{ maxWidth: "100%", height: "auto" }} />
                    )}
                </div>
            )}
        </div>
    );
}

✅ Handles text and image responses dynamically!

📌 Step 3: Running the System

1️⃣ Start Spring Boot Backend

mvn spring-boot:run

	•	Available at http://localhost:8080/api/rag/query.

2️⃣ Start Next.js Frontend

npm run dev

	•	Available at http://localhost:3000.

📌 Final Result

1️⃣ Text Query Example

Input: "What is Culex?"
Response: "Culex is a genus of mosquitoes, some species of which are vectors for diseases."

2️⃣ Image Query Example

Input: "Image of a mosquito"
Response: 🖼️ DALL·E generated mosquito image.

📌 Summary

Component	Technology	Purpose
Frontend	React, Next.js	UI for user queries & dynamic display of text/images
Backend	Spring Boot	Handles API requests, processes queries
LLM	OpenAI GPT-4	Generates text responses
Image Gen	DALL·E	Generates AI-based images
Data Retrieval	FAISS (Optional)	Retrieves relevant content

📌 Next Steps
	1.	Integrate FAISS for retrieving real-world documents.
	2.	Add real-time streaming responses with WebSockets.
	3.	Enhance UI using Tailwind CSS or Material UI.



Here are 10 high-performance code snippets that handle latency reduction, complex data manipulation, and real-time updates in your Spring Boot + React RAG system.

📌 1️⃣ Async Query Handling in Spring Boot

💡 Reduces response latency by executing AI calls asynchronously.

import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.stereotype.Service;
import java.util.concurrent.CompletableFuture;

@EnableAsync
@Service
public class AsyncRagService {

    @Async
    public CompletableFuture<String> processQueryAsync(String query) {
        String response = callOpenAI(query); // Simulating external API call
        return CompletableFuture.completedFuture(response);
    }

    private String callOpenAI(String query) {
        // Simulate API call
        return "Processed query: " + query;
    }
}

✅ Improves request handling by executing AI calls without blocking the main thread.

📌 2️⃣ WebSockets for Real-Time AI Responses in Spring Boot

💡 Streams AI responses instead of waiting for full completion.

import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@RestController
@RequestMapping("/api/stream")
public class StreamingController {

    private final ExecutorService executor = Executors.newCachedThreadPool();

    @GetMapping("/query")
    public SseEmitter streamAIResponse(@RequestParam String query) {
        SseEmitter emitter = new SseEmitter();
        executor.execute(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    emitter.send("Chunk " + i + ": AI processing...");
                    Thread.sleep(1000);
                }
                emitter.complete();
            } catch (IOException | InterruptedException e) {
                emitter.completeWithError(e);
            }
        });
        return emitter;
    }
}

✅ Reduces perceived latency by sending AI-generated chunks in real-time.

📌 3️⃣ React: Handling Streaming Responses with WebSockets

💡 Receives streamed AI responses in the frontend.

import { useEffect, useState } from "react";

export default function StreamingResponse() {
    const [messages, setMessages] = useState([]);

    useEffect(() => {
        const eventSource = new EventSource("/api/stream/query?query=Hello");
        eventSource.onmessage = (event) => {
            setMessages((prev) => [...prev, event.data]);
        };
        eventSource.onerror = () => eventSource.close();
        return () => eventSource.close();
    }, []);

    return (
        <div>
            <h1>Streaming AI Response</h1>
            {messages.map((msg, index) => <p key={index}>{msg}</p>)}
        </div>
    );
}

✅ Dynamically updates the UI as AI responses stream in.

📌 4️⃣ Efficient Vector Search with FAISS in Java

💡 Finds the most relevant document with optimized FAISS search.

import faiss.*;

public class FaissSearch {
    private final IndexFlatL2 faissIndex;

    public FaissSearch(int dimension) {
        this.faissIndex = new IndexFlatL2(dimension);
    }

    public int search(float[] queryVector) {
        int k = 1; // Retrieve top 1 result
        float[] distances = new float[k];
        int[] indices = new int[k];

        faissIndex.search(1, queryVector, k, distances, indices);
        return indices[0];
    }
}

✅ Speeds up retrieval by using FAISS optimized vector search.

📌 5️⃣ Caching AI Responses with Redis

💡 Reduces latency by caching AI-generated responses.

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class CacheService {
    private final StringRedisTemplate redisTemplate;

    public CacheService(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public String getCachedResponse(String query) {
        return redisTemplate.opsForValue().get(query);
    }

    public void cacheResponse(String query, String response) {
        redisTemplate.opsForValue().set(query, response);
    }
}

✅ Prevents redundant AI calls and improves performance.

📌 6️⃣ Debouncing API Calls in React

💡 Reduces API request spam when a user types fast.

import { useState, useEffect } from "react";
import axios from "axios";

export default function SearchBar() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState(null);

    useEffect(() => {
        const delayDebounce = setTimeout(() => {
            if (query) fetchResults(query);
        }, 500);

        return () => clearTimeout(delayDebounce);
    }, [query]);

    const fetchResults = async (query) => {
        const { data } = await axios.post("/api/rag/query", { query });
        setResults(data);
    };

    return (
        <input
            type="text"
            placeholder="Type a query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
        />
    );
}

✅ Prevents unnecessary API calls while the user types.

📌 7️⃣ React Query for Data Fetching with Caching

💡 Efficiently fetches and caches AI responses.

import { useQuery } from "react-query";
import axios from "axios";

const fetchAIResponse = async (query) => {
    const { data } = await axios.post("/api/rag/query", { query });
    return data;
};

export default function QueryComponent({ query }) {
    const { data, isLoading } = useQuery(["aiResponse", query], () => fetchAIResponse(query), { enabled: !!query });

    return isLoading ? <p>Loading...</p> : <p>{data}</p>;
}

✅ Automatically caches API results and avoids redundant calls.

📌 8️⃣ Rate-Limiting API Requests in Spring Boot

💡 Prevents excessive API usage.

import org.springframework.stereotype.Component;
import io.github.bucket4j.*;

import java.time.Duration;

@Component
public class RateLimiter {
    private final Bucket bucket;

    public RateLimiter() {
        Bandwidth limit = Bandwidth.simple(5, Duration.ofMinutes(1));
        this.bucket = Bucket4j.builder().addLimit(limit).build();
    }

    public boolean tryConsume() {
        return bucket.tryConsume(1);
    }
}

✅ Limits users to 5 API calls per minute.

📌 9️⃣ Error Handling for OpenAI API Calls in Spring Boot

💡 Handles failures gracefully.

public String safeCallOpenAI(String query) {
    try {
        CompletionRequest completionRequest = CompletionRequest.builder()
            .model("gpt-4")
            .prompt(query)
            .maxTokens(100)
            .build();

        return openAiService.createCompletion(completionRequest).getChoices().get(0).getText();
    } catch (Exception e) {
        return "Error: AI service is currently unavailable.";
    }
}

✅ Prevents system crashes when OpenAI API fails.

📌 🔟 Dynamic Image Rendering in React

💡 Handles both text and images dynamically.

export default function AIResponse({ data }) {
    return (
        <div>
            {data.type === "text" ? (
                <p>{data.content}</p>
            ) : (
                <img src={data.content} alt="Generated Image" width="300" />
            )}
        </div>
    );
}

✅ Seamlessly switches between AI text and images.

📌 Conclusion

These 10 snippets enhance:
	•	Latency reduction (WebSockets, caching, async calls).
	•	Real-time AI responses (Streaming API, React Query).
	•	Efficiency in data retrieval (FAISS search, rate limiting).
	•	Dynamic UI updates (Debouncing, text/image handling).

🚀 Would you like real-time AI text + speech output next? 🎙️


