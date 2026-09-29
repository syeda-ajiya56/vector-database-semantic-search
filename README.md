# Vector Database and Semantic Search

## 1. Project Overview

This project is a small practical experiment demonstrating how a vector database can store embeddings and retrieve semantically related information.

The project uses a fictional university FAQ dataset containing 10 short pieces of information. Each piece of information is converted into an embedding using the Google Gemini embedding model and stored in a local Chroma vector database.

Natural-language questions are also converted into embeddings. Chroma then performs similarity search and returns the most relevant stored information.

The goal is to understand how embeddings, vector databases, metadata, and semantic search work together in a Generative AI application.

---

## 2. What Is a Vector Database?

A vector database is a database designed to store and search numerical vector representations of information.

An embedding model converts text into a vector containing numerical values. A vector database stores these vectors and can efficiently search for vectors that are similar to a query vector.

This makes vector databases useful for applications such as:

- Semantic search
- AI knowledge bases
- Recommendation systems
- Retrieval-Augmented Generation (RAG)
- Document search

---

## 3. Traditional Database vs Vector Database

A traditional database is commonly used for structured queries.

For example:

> Find all products where price is less than 5000.

A vector database is useful for similarity-based queries.

For example:

> Find information related to financial support for students.

Instead of requiring an exact keyword match, vector search compares the semantic representation of the query with stored vectors.

---

## 4. Technologies Used

- Python
- Google Gemini API
- `google-genai`
- ChromaDB
- `python-dotenv`
- Gemini embedding model: `gemini-embedding-001`

---

## 5. How the System Works

The basic flow is:

```text
University FAQ Information
          ↓
    Gemini Embedding Model
          ↓
     Numerical Vectors
          ↓
    Chroma Vector Database
          ↓
    Natural-Language Question
          ↓
    Question Embedding
          ↓
     Similarity Search
          ↓
    Most Relevant Results