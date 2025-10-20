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

## Code Structure & Improvements

### Data_Visualization.py
- **Purpose**: Exploratory data analysis.
- **Changes**:
  - Used `Counter` for tag frequency analysis.
  - Filtered `region` counts to cities only for precision.
- **Debugging**: Added `dropna()` for `tags` to handle missing data; validated counts with `df.describe()`.

### Pinecone_Upload.py
- **Purpose**: Efficiently embed and upsert dataset to Pinecone.
- **Changes**:
  - Fallback to `description[:1000]` if `semantic_text` is absent.
  - Implemented `chunked` generator for memory-efficient batching.
  - Added `time.sleep(0.2)` to prevent rate limiting (observed at >50 req/s).
- **Debugging**: Logged batch sizes; asserted embedding dims match `VECTOR_DIM`.

### Test_Hybrid.py
- **Purpose**: Core RAG pipeline with interactive chat.
- **Changes**:
  - **Retry Logic**: 3 retries for model loading with `ReadTimeout/ConnectionError` handling.
  - **Index Handling**: Safely parses `pc.list_indexes()`; catches 409 errors with detailed logging (e.g., region mismatch).
  - **Prompt Design**: Structured with few-shot examples; grouped matches by type; added `best_time_to_visit` from dataset.
  - **Query Detection**: Keyword-based with dynamic temperature (e.g., 0.15 for timing, 0.35 for food).
  - **Graph Query**: Limited to 40 results; truncated descriptions to 200 chars.
  - **Error Fallbacks**: Returns empty lists on query failures; exits only after retries.
- **Debugging**:
  - Logged prompt length to avoid token limits (600 max).
  - Monitored `len(matches)` to optimize `TOP_K`.
  - Used `try-except` for all external API calls.

## Debugging Techniques

| Issue | Technique | Value |
|-------|-----------|-------|
| **Network Timeouts** | 3 retries with 5s delays for model loading. | Handles Hugging Face mirror instability; logs attempts. |
| **Pinecone 409 Conflict** | Caught exception; logged HTTP headers/body. | Identifies region/env issues; prevents crashes. |
| **Rate Limiting** | 32-item batches with 0.2s delays. | Scales to thousands of upserts; `tqdm` tracks progress. |
| **Prompt Overflow** | Truncated descriptions; grouped matches. | Stays under Gemini's 600-token limit; improves clarity. |
| **NaN in Dataset** | `dropna()` and fallback logic. | Ensures valid embeddings; quick validation via `df.info()`. |
| **Graph Overfetch** | Cypher `LIMIT 40`; 200-char desc cap. | Keeps query time <100ms; prevents memory issues. |

Aspect,Original Version,Improved Version,Key Improvement
Dependencies & Models,"- OpenAI (text-embedding-3-small for embeddings, gpt-4o-mini for chat).
- Pinecone, Neo4j.
- Relies on API keys for all external services.","- SentenceTransformer (all-MiniLM-L6-v2 for embeddings).
- Google Gemini (gemini-2.0-flash for chat).
- Pinecone, Neo4j unchanged.
- Adds requests for timeout handling, os/json for local file ops.","Cost/Offline Efficiency: Switches to local embeddings (download once, no per-query API calls; ~384-dim vs. original's 1536-dim for lighter compute). Gemini may reduce costs vs. OpenAI. Enables partial offline use post-model load."
Initialization,"- Direct client init (OpenAI, Pinecone, Neo4j).
- Index creation if missing (hard-coded region us-east1-gcp).
- No retries or local data.","- Gemini config first.
- Retries (3x with 5s sleep) for model load, handling timeouts/ConnectionErrors.
- Safer index listing (handles dict/list response formats) + creation (uses config.PINECONE_ENV for region).
- Optional local JSON dataset load for extra metadata (e.g., best_time_to_visit).","Robustness: Adds fault-tolerant model loading (prevents crashes on flaky networks). Flexible index handling avoids errors if API responses vary. Local dataset (~O(1) lookup) enriches context without DB hits, reducing latency."
Embedding Function,- embed_text: Calls OpenAI API synchronously.,"- embed_text: Uses local model.encode() (batch=1, to list).",Performance: Local inference is faster (~10-50ms vs. API latency) and cheaper (no tokens billed). Batch-ready for future scaling.
Pinecone Query,"- pinecone_query: Embeds via API, queries with include_metadata=True, include_values=False.
- Prints debug len(res[""matches""]).
- Assumes success.","- pinecone_query: Same query params, but wraps in try-except (returns [] on error).
- No debug print.",Error Resilience: Graceful degradation (empty list on failure) prevents full crashes. Cleaner (no debug spam).
Graph Context Fetch,"- fetch_graph_context: Loops per node_id, queries neighbors (depth=1, LIMIT 10 per node).
- Includes labels, full desc[:400].
- Prints debug len(facts).","- fetch_graph_context: Single batched Cypher query (WHERE n.id IN $ids, LIMIT 40 total).
- No labels, desc[:200].
- Early return if no IDs.",Efficiency/Scalability: Batched query reduces Neo4j round-trips (O(1) vs. O(K) for TOP_K=5). Tighter limits prevent token bloat. Shorter descs keep prompts concise.
Query Type Detection,- None (fixed temperature=0.2).,"- detect_query_type: Keyword-based classifier (e.g., ""itinerary"" if ""plan"" in query).
- Maps to temperature (0.15-0.35).","Response Quality: Dynamic temperature tailors creativity (e.g., more varied for ""activity"" queries). Simple but effective for domain-specific tuning without full classification."
Prompt Building,"- build_prompt: Basic system msg + vec snippets (id/name/type/score/city) + graph triples.
- User msg with TOP_K=5 limit, suggests 2-3 steps, cites IDs.
- Returns list of dicts for OpenAI.","- build_prompt: Groups matches by type (City/Attraction/etc.), formats with local dataset (e.g., best_time).
- Graph as simple triples.
- Adds few-shot examples for style.
- Structured sections (Instructions, Query, Matches, Facts, Examples).
- Returns single string for Gemini.","Prompt Engineering: Grouping + local integration makes context more scannable/actionable (e.g., ""Best time: Feb-May""). Few-shots guide concise, itinerary-focused outputs. Reduces hallucination by emphasizing citations/nearby recs. Overall: More structured for better LLM adherence."
Chat Generation,"- call_chat: OpenAI chat.completions.create (max_tokens=600, temp=0.2 fixed).","- call_chat: Gemini generate_content (max_output_tokens=600, configurable temp).","Flexibility: Temp param enables query-adaptive responses. Gemini's config aligns closely, but error handling added (returns error msg)."
Interactive Loop,"- Simple while loop: Input, query/embed/graph/prompt/chat, print answer.
- Exit on ""exit""/""quit""/empty.
- TOP_K=5 fixed.","- Enhanced loop: Skips empty input, detects type for temp, extracts IDs for graph.
- Friendlier prints (""Hi, I am your Vietnam Travel Assistant"").",UX/Logic: Smarter input handling + type-aware generation improves flow. Cleaner output formatting.
Overall Structure,"- ~150 lines, basic sections (Config, Init, Helpers, Chat).
- Hard-coded prints/debug.
- No local files.","- ~200 lines, same sections + dataset load.
- More comments, init logs (e.g., ""[INIT] Loaded dataset..."").
- Configurable via config.py (e.g., region).","Maintainability: Better logging for debugging/deploy. Modular (e.g., temp mapping). Prepares for extensions like dataset updates."
