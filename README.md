# Vietnam Travel AI Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

**Vietnam Travel AI Assistant** is a hybrid retrieval-augmented generation (RAG) system for personalized travel recommendations in Vietnam. It integrates semantic search (Pinecone), graph-based relationships (Neo4j), and natural language generation (Google Gemini API) to provide concise, itinerary-style responses based on a dataset of 360 travel entities (cities, attractions, hotels, activities).

### Key Features
- **Semantic Search**: Uses `all-MiniLM-L6-v2` SentenceTransformer for 384-dimensional embeddings, stored in Pinecone with cosine similarity.
- **Graph Context**: Neo4j captures relationships (e.g., `City -HAS-> Attraction`) for contextual recommendations.
- **LLM Responses**: Gemini 2.0 Flash generates user-friendly answers with few-shot prompting.
- **Query Detection**: Adjusts response style/temperature based on query type (e.g., itinerary, hotel, food).
- **Data Visualization**: Generates stats and visuals (e.g., entity type pie chart) for dataset exploration.
- **Robust Upload**: Batch uploads to Pinecone with rate limiting and progress tracking.
- **Error Handling**: Retries for network issues, safe index creation, and fallback mechanisms.

## Dataset

The `vietnam_travel_dataset.json` contains 360 entities:
- **Entity Types**:
  | Type       | Count |
  |------------|-------|
  | Attraction | 150   |
  | Hotel      | 100   |
  | Activity   | 100   |
  | City       | 10    |

- **City Distribution**: 35 entities each for Hanoi, Ha Long Bay, Sapa, Hue, Hoi An, Da Nang, Nha Trang, Da Lat, Ho Chi Minh City, Mekong Delta.
- **Top Tags**:
  | Tag        | Count |
  |------------|-------|
  | stay       | 100   |
  | experience | 100   |
  | beach      | 33    |
  | romantic   | 29    |
  | mountain   | 25    |
- **Regions (Cities)**:
  | Region           | Count |
  |------------------|-------|
  | Southern Vietnam | 4     |
  | Northern Vietnam | 3     |
  | Central Vietnam  | 3     |

**Visualization**: Pie chart of entity types (via Matplotlib):
- Attractions: 41.7%
- Hotels: 27.8%
- Activities: 27.8%
- Cities: 2.8%

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Embedding      │───▶│   Pinecone      │
│   (Interactive) │    │   (SentenceT)    │    │   (Vector DB)   │
└──────────┬──────┘    └──────────┬──────┘    └──────────┬──────┘
           │                      │                       │
           ▼                      ▼                       ▼
┌──────────┴──────────┐ ┌──────────┴──────────┐ ┌──────────┴──────────┐
│   Query Type        │ │   Semantic Matches   │ │   Graph Facts       │
│   Detection         │ │   (Top-K)            │ │   (Neo4j Cypher)    │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                      │                       │
           └──────────────────────┼───────────────────────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │   Prompt     │
                           │   Builder    │
                           │   (Few-Shot) │
                           └──────┬───────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │   Gemini     │
                           │   (Response) │
                           └──────────────┘
```

## Setup

### Prerequisites
- Python 3.8+
- API Keys: Pinecone, Google Gemini, Neo4j (Aura or local instance).
- Dependencies: Install via `pip install pinecone-client sentence-transformers neo4j google-generativeai tqdm torch pandas matplotlib`.

### Configuration
Create `config.py`:
```python
NEO4J_URI = "neo4j+s://your-instance.neo4j.io"  # Or bolt://localhost:7687
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-password"

PINECONE_BATCH__SIZE = 32
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384
PINECONE_ENV = "us-east-1"

GEMINI_API_KEY = "your-gemini-key"
```

### Installation
1. Clone repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Place `vietnam_travel_dataset.json` in project root.
4. Ingest Neo4j relationships (use separate Cypher script, not provided).
5. Run Pinecone upload: `python Pinecone_Upload.py`

## Usage

### 1. Data Visualization
`python Data_Visualization.py`
- Outputs dataset stats (entity counts, tag frequencies, regional breakdown).
- Displays pie chart of entity types.
- **Debug Tip**: Check for NaNs in `tags` using `df.info()` to ensure accurate counts.

### 2. Pinecone Upload
`python Pinecone_Upload.py`
- Creates Pinecone index (`vietnam-travel`) if missing.
- Upserts 360 entities in batches of 32, with 0.2s delays to avoid rate limits.
- Uses `tqdm` for progress tracking.
- **Sample Output**:
  ```
  Creating managed index: vietnam-travel
  Preparing to upsert 360 items to Pinecone...
  Uploading batches: 100%|██████████| 12/12 [00:26<00:00,  2.17s/it]
  All items uploaded successfully.
  ```
  <img width="1600" height="815" alt="image" src="https://github.com/user-attachments/assets/1f9d5048-13cb-45a8-b297-cc925989a031" />

- **Debug Tip**: Verify embedding dimensions (`len(emb) == 384`) and log batch sizes.

### 3. Interactive Chat
`python Test_Hybrid.py`
- Loads embedding model with 3 retries (5s delays for network issues).
- Safely handles Pinecone index creation (ignores 409 conflicts).
- Interactive loop for queries like "budget hotels in Hanoi" or "beach activities in Da Nang".
- **Sample Interaction**:
  <img width="1535" height="763" alt="image" src="https://github.com/user-attachments/assets/5dcf8645-1ce1-4ebc-9e92-99bcab8b0296" />

- **Debug Tip**: Print `prompt_text[:500]` to check for token overflow; adjust `TOP_K` for larger datasets.

## Debugging Techniques

| Issue | Technique | Value |
|-------|-----------|-------|
| **Network Timeouts** | 3 retries with 5s delays for model loading. | Handles Hugging Face mirror instability; logs attempts. |
| **Pinecone 409 Conflict** | Caught exception; logged HTTP headers/body. | Identifies region/env issues; prevents crashes. |
| **Rate Limiting** | 32-item batches with 0.2s delays. | Scales to thousands of upserts; `tqdm` tracks progress. |
| **Prompt Overflow** | Truncated descriptions; grouped matches. | Stays under Gemini's 600-token limit; improves clarity. |
| **NaN in Dataset** | `dropna()` and fallback logic. | Ensures valid embeddings; quick validation via `df.info()`. |
| **Graph Overfetch** | Cypher `LIMIT 40`; 200-char desc cap. | Keeps query time <100ms; prevents memory issues. |


## 🔍 Comparison Summary

| **Aspect** | **Original Version** | **Improved Version** | **Key Improvement** |
|-------------|----------------------|----------------------|----------------------|
| **Dependencies & Models** | - OpenAI (`text-embedding-3-small` for embeddings, `gpt-4o-mini` for chat).<br>- Pinecone, Neo4j.<br>- Relies on API keys for all external services. | - SentenceTransformer (`all-MiniLM-L6-v2` for embeddings).<br>- Google Gemini (`gemini-2.0-flash` for chat).<br>- Pinecone, Neo4j unchanged.<br>- Adds `requests` for timeout handling, `os`/`json` for local file ops. | **Cost/Offline Efficiency:** Switches to local embeddings (download once, no per-query API calls; ~384-dim vs. original’s 1536-dim). Gemini may reduce costs vs. OpenAI and enables partial offline use. |
| **Initialization** | - Direct client init (OpenAI, Pinecone, Neo4j).<br>- Index creation if missing (hard-coded region `us-east1-gcp`).<br>- No retries or local data. | - Gemini config first.<br>- Retries (3x with 5s sleep) for model load, handling timeouts/ConnectionErrors.<br>- Safer index listing (handles dict/list formats) + creation using `config.PINECONE_ENV`.<br>- Optional local JSON dataset load for extra metadata. | **Robustness:** Adds fault-tolerant model loading, flexible index handling, and local dataset integration for faster contextual lookups. |
| **Embedding Function** | `embed_text`: Calls OpenAI API synchronously. | `embed_text`: Uses local `model.encode()` (batch=1, converts to list). | **Performance:** Local inference is faster (~10–50ms vs. API latency) and cheaper (no tokens billed). Batch-ready for scaling. |
| **Pinecone Query** | - Embeds via API and queries Pinecone (`include_metadata=True`).<br>- Prints debug output.<br>- Assumes success. | - Same query params but wrapped in `try-except` (returns empty list on error).<br>- No debug print. | **Error Resilience:** Graceful degradation prevents crashes. Cleaner execution without debug spam. |
| **Graph Context Fetch** | - Loops per `node_id`, separate queries (depth=1, LIMIT 10).<br>- Includes labels, full desc[:400]. | - Single batched Cypher query (`WHERE n.id IN $ids`, LIMIT 40).<br>- No labels, desc[:200].<br>- Early return if no IDs. | **Efficiency/Scalability:** Batched query reduces Neo4j round-trips (O(1) vs. O(K)). Shorter context avoids token bloat. |
| **Query Type Detection** | None (fixed temperature = 0.2). | Adds `detect_query_type`: simple keyword classifier (e.g., “plan” → itinerary).<br>Maps to temperature (0.15–0.35). | **Response Quality:** Dynamic temperature tailors creativity per query type for domain-specific optimization. |
| **Prompt Building** | - Basic system + vector snippet prompt.<br>- Lists ID/name/type/score/city.<br>- Returns list of dicts for OpenAI. | - Groups matches by type (City/Attraction/etc.), integrates local metadata (e.g., best time to visit).<br>- Graph triples simplified.<br>- Adds few-shot examples.<br>- Structured sections: **Instructions**, **Query**, **Matches**, **Facts**, **Examples**.<br>- Returns single formatted string for Gemini. | **Prompt Engineering:** Structured, grouped, and metadata-enriched prompts reduce hallucinations and improve travel relevance. |
| **Chat Generation** | Calls `OpenAI.chat.completions.create` (max_tokens=600, temp=0.2 fixed). | Uses `Gemini.generate_content` (max_output_tokens=600, configurable temp). | **Flexibility:** Adjustable temperature per query. Adds error handling to return safe fallbacks. |
| **Interactive Loop** | - Simple `while` loop (input → query → output).<br>- Exits on `"exit"` / `"quit"`. | - Skips empty input.<br>- Detects query type and adjusts temperature.<br>- Extracts IDs for graph fetch.<br>- Adds friendly intro (“Hi, I am your Vietnam Travel Assistant”). | **UX/Logic:** Smarter flow and friendlier user interaction. |
| **Overall Structure** | - ~150 lines.<br>- Basic config/init/helpers.<br>- Hard-coded prints, no local files. | - ~200 lines.<br>- Adds dataset load, structured logs, and config flexibility (e.g., region). | **Maintainability:** Better logging, modularity, and easier extension (e.g., dataset updates, async support). |

---


### ✅ Improvements Achieved
- **Reliability:** Handles timeouts, retries, and API inconsistencies gracefully.  
- **Efficiency:** Local embeddings eliminate API cost and latency.  
- **Scalability:** Batched Neo4j queries and modular design.  
- **Quality:** Smarter prompting and adaptive temperatures produce more relevant, concise travel responses.  
- **Extensibility:** Easy to plug in local datasets or switch model providers.


