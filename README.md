# UltraDoc Intelligence RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed for logistics document processing. This system extracts structured information from documents, enables semantic question answering, and implements robust guardrails to prevent hallucinations.

## Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

**Data Layer** (`src/core/data/`)
- **Document Processor**: Extracts raw text from PDF, DOCX, and TXT files using PyPDF and pdfplumber
- **LLM Structured Extractor**: Uses GPT-4.1 to normalize document text into canonical JSON format
- **Chunker**: Breaks structured JSON into field-level semantic chunks for granular retrieval
- **Vector Store**: In-memory FAISS index for efficient similarity search
- **Schemas**: Pydantic models for data validation

**Services Layer** (`src/core/services/`)
- **Embedding Service**: Generates embeddings using OpenAI's text-embedding-3-large model
- **Retriever**: Orchestrates embedding generation and vector similarity search
- **Answer Generator**: Generates grounded answers using GPT-4.1 with conversational memory

**Evaluator Layer** (`src/core/evaluator/`)
- **Guardrails**: Two-stage validation to prevent hallucinations
- **Confidence Scorer**: Computes multi-factor confidence scores for answers

**State Layer** (`src/core/state/`)
- **App State**: Manages shared in-memory state across API endpoints
- **Memory Manager**: Handles short-term conversational memory (last 10 interactions)

### API Layer

The system exposes a FastAPI backend with the following endpoints:
- `/upload` - Document upload and indexing
- `/ask` - Question answering with confidence scoring
- `/extract` - Structured data extraction
- `/clear_memory` - Memory reset

### Frontend

Streamlit-based UI providing document upload, Q&A interface, and structured extraction visualization.

## Chunking Strategy

The system uses **field-level semantic chunking** optimized for structured documents. Raw text is first converted to structured JSON using GPT-4.1, then each field becomes an independent chunk with section metadata preserved.

**Example:**
```
Original: { "shipment_info": { "origin": "Mumbai", "destination": "Delhi" } }

Chunks:
- "Section: Shipment Info | origin: Mumbai"
- "Section: Shipment Info | destination: Delhi"
```

### Why Field-Level Indexing?

Indexing entire sections as single embeddings diluted semantic specificity.
By embedding each key-value pair independently, similarity scores improved significantly for targeted queries (e.g., "What is the mailing address?").

This approach mirrors document indexing strategies used in Elasticsearch and modern vector databases. It provides precision (queries retrieve only relevant fields), scalability (works with varying document structures), and flexibility (handles nested objects naturally). The tradeoff is requiring an upfront LLM call for structure extraction, but this enables much more accurate retrieval downstream.

## Retrieval Method

The system implements **semantic similarity-based retrieval** using FAISS:

1. **Query Embedding**: User question is embedded using text-embedding-3-large (3072 dimensions)
2. **Vector Search**: FAISS performs cosine similarity search against indexed chunks
3. **Top-K Selection**: Retrieves top 4 most similar chunks
4. **Threshold Filtering**: Only chunks with similarity >= 0.30 are retained

The similarity threshold of 0.30 was chosen to accommodate variance in question phrasing while filtering out clearly irrelevant chunks. This is intentionally permissive because the guardrails layer provides additional validation.

## Guardrails Approach

The system implements a **two-stage guardrail mechanism** to prevent hallucinations:

**Stage 1 - Retrieval Validation** (before answer generation):
- At least one chunk must be retrieved
- Maximum similarity score must exceed 0.30
- If validation fails, returns "Not found in document" without calling the LLM

**Stage 2 - Confidence Validation** (after answer generation):
- Final confidence score must exceed 0.45
- If too low, returns "I have low confidence in the generated answer"

The guardrails are conservative by design, erring on the side of saying "I don't know" rather than hallucinating. This is critical for enterprise applications where incorrect information can have serious consequences.

## Confidence Scoring Method

The system computes a **weighted multi-factor confidence score** combining three signals:

1. **Retrieval Score (50% weight)** - Maximum cosine similarity of retrieved chunks (0 if below threshold)
2. **Chunk Agreement Score (30% weight)** - Number of supporting chunks (normalized: 3+ chunks = 1.0)
3. **Answer Coverage Score (20% weight)** - Word overlap between answer and retrieved context

**Formula:** `confidence = 0.5 * retrieval_score + 0.3 * agreement_score + 0.2 * coverage_score`

Answers with confidence < 0.45 are rejected by the guardrails. The weights prioritize retrieval quality while considering supporting evidence and grounding.

## Failure Cases

Based on testing, the system has the following known limitations:

1. **Ambiguous References** - Questions with pronouns or vague references may fail. Example: "What is the delivery date?" when multiple shipments exist.

2. **Cross-Field Reasoning** - Questions requiring synthesis across unrelated sections struggle. Example: "What is the total value of shipments from Mumbai?" requires aggregating multiple chunks.

3. **Numerical Calculations** - The system only retrieves existing values, cannot perform arithmetic. Example: "What is the difference between origin and destination distances?" will fail.

4. **Implicit Information** - Questions about unstated information fail correctly. Example: "Is this shipment delayed?" when only dates are provided.

5. **Very Long Documents** - In-memory FAISS index may cause issues with documents > 100 pages, leading to potential crashes or slow performance.

## Improvement Ideas

### Short-term Improvements

1. **Hybrid Chunking**: Combine field-level chunks with section-level chunks to capture broader context
2. **Reranking**: Add a cross-encoder reranking step after initial retrieval to improve precision
3. **Query Expansion**: Use LLM to generate alternative phrasings of user questions before retrieval
4. **Caching**: Cache embeddings and LLM responses to reduce API costs and latency

### Medium-term Improvements

1. **Multi-hop Reasoning**: Implement iterative retrieval for questions requiring multiple pieces of information
2. **Entity Extraction**: Extract and index entities (shipment IDs, locations, dates) separately for exact matching
3. **Metadata Filtering**: Allow filtering by document sections or field types before semantic search
4. **Conversation Summarization**: Compress memory using summarization instead of storing raw interactions

### Long-term Improvements

1. **Fine-tuned Embeddings**: Train domain-specific embeddings on logistics documents
2. **Graph-based Retrieval**: Model document structure as a graph to preserve relationships
3. **Agentic Workflow**: Allow the system to decide when to retrieve, calculate, or refuse to answer
4. **Persistent Storage**: Move from in-memory to persistent vector database (Pinecone, Weaviate)
5. **Evaluation Framework**: Build automated test suite with ground truth Q&A pairs

## How to Run Locally

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ultradoc-intelligence-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI backend**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

5. **In a separate terminal, start the Streamlit frontend**
   ```bash
   source venv/bin/activate  # Activate venv again
   streamlit run streamlit_app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Enter your OpenAI API key in the sidebar
   - Upload a document and start asking questions

### Note on Docker
Future production deployment would include Docker containerization and environment-based configuration management.

## Project Structure

```
ultradoc-intelligence-rag/
├── src/
│   ├── api/                    # FastAPI endpoints
│   ├── config/                 # Configuration settings
│   └── core/
│       ├── data/              # Data processing and storage
│       ├── evaluator/         # Guardrails and confidence scoring
│       ├── services/          # Business logic services
│       └── state/             # Application state management
├── streamlit_app.py           # Frontend UI
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration

Key parameters can be adjusted in `src/config/settings.py`:

```python
TOP_K_RETRIEVAL = 4              # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.30      # Minimum similarity for retrieval
MIN_CONFIDENCE_SCORE = 0.45      # Minimum confidence to return answer
MAX_SHORT_TERM_MEMORY = 10       # Conversation history length
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"
```

## Author

**Sagnik Mukherjee**

GitHub: [github.com/sagnik0712mukherjee](https://github.com/sagnik0712mukherjee)

---

This project was created as an assignment submission.
