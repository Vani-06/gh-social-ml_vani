# ML Work Assignments: GitHub Repo Recommender Pipeline

This document contains actionable task descriptions (designed like GitHub Issues or Jira Tickets) that you can distribute to each of your 7 ML engineers.

---

## 📋 Merged PR Log

### ✅ PR #3: `feat/ingestion-pipeline` → `main` (by @ramsidhartha)

Added the full **Ingestion Pipeline** module that evaluates repository quality after acquisition. This implements the core of **Ticket 1** (Quality Modeler & Evaluator).

**New files added:**
- `ingestion/pipeline.py` — Main `ingest_repository()` and `ingest_batch()` orchestration
- `ingestion/features.py` — Documentation quality scorer, activity scorer, code health scorer, trend velocity scorer, tag extraction, structured summary builder
- `ingestion/classification.py` — Taxonomy category classifier (maps repos into categories like AI/ML, DevOps, Frontend, etc.)
- `ingestion/corpus.py` — `CorpusStore` for tracking ingested corpus and `dynamic_cluster_discovery()` for topology analysis
- `ingestion/result.py` — `IngestionResult` dataclass with quadrant classification (🔥 Viral Rockets, 💎 Hidden Gems, etc.)
- `ingestion/config.py` — Gate thresholds (`GATE_APPROVAL_THRESHOLD`, `MIN_STARS_PREFILTER`, `MIN_README_PREFILTER`)
- `ingestion_engine.py` — Top-level engine integrating acquisition → ingestion

**Modified files:**
- `acquisition/github_discovery.py` — Migrated from REST to pure GraphQL
- `acquisition/github_graphql_client.py` — Extended GraphQL client
- `acquisition/graphql_queries.py` — Simplified queries
- `acquisition/repository_enricher.py` — Updated enrichment logic
- `.gitignore` — Added Python cache exclusions

**Deleted files:**
- `acquisition/github_client.py` — REST client removed (fully replaced by GraphQL)

---

### ✅ `feat/storage-connector` (in progress)

Added the **PostgreSQL Database Connector** for persisting enriched repositories. Supports both local PostgreSQL and cloud-hosted Supabase.

**New files:**
- `database/connector.py` — `PostgreSQLConnector` with SSL auto-detection, connection verification, schema migrations, upsert logic
- `database/__init__.py` — Package init
- `.env.example` — Reference environment variable template

---

## 🎟️ Ticket 1: Quality Modeler & Evaluator (Engineer 1)

### Objective
Implement a pipeline stage that scores the quality of raw ingested GitHub repositories and generates a composite `quality_score` to filter out spam, config dumps, and low-quality codebases.

### Status: 🟡 Partially Complete (via PR #3)

**What's done:**
- Documentation Quality Scorer (`ingestion/features.py` → `score_documentation()`)
- Activity Scorer (`ingestion/features.py` → `activity_score()`)
- Code Health Scorer (`ingestion/features.py` → `score_code_health()`)
- Trend Velocity Scorer (`ingestion/features.py` → `trend_velocity()`)
- Composite quality gate in `ingestion/pipeline.py` (50/50 blend of doc quality + code health)
- Basic quality filter in `main.py` (README length check)

**What's remaining:**
- Composite Quality Score Model: Consolidate into a single `calculate_quality(repo_payload: dict) -> Tuple[float, dict]` API that combines all sub-scores
- Persist `quality_score` and `quality_metrics` columns in the database
- Unit tests validating that empty/useless repos score < 0.2 and highly-active open-source repos score > 0.8

### Input Data
Raw repository metadata and README text from the database `Repo` table.

### Key Tasks
1. **Documentation Quality Scorer** ✅: Parse the README markdown and calculate a score based on section coverage (e.g., presence of Install, Usage, License, Examples).
2. **Activity Scorer** ✅: Calculate score based on commit activity frequency (commits per week), open/closed issue ratio, and commit recency.
3. **Code Health Scorer** ✅: Compute ratio of file size to description length, open issue volume relative to stargazers, and primary language dominance.
4. **Trend Velocity Scorer** ✅: Compute exponential growth factors for stars and forks over 3d, 7d, and 30d windows.
5. **Composite Quality Score Model** 🔲: Build a heuristic scoring model or a simple regression classifier mapping the sub-scores to a single value $S_{quality} \in [0.0, 1.0]$.
6. **Pipeline Integration** 🟡: Write a Python module that takes a batch of raw repositories, computes the score, and saves the output to the database.

### Outputs & APIs
- A python function: `calculate_quality(repo_payload: dict) -> Tuple[float, dict]` returning the score and a metadata payload.

### Definition of Done (DoD)
- Unit tests validating that empty/useless repos score $< 0.2$ and highly-active open-source repos score $> 0.8$.
- Script successfully updates the database columns `quality_score` and `quality_metrics` for 100 sample repos.

---

## 🎟️ Ticket 2: Repo Representation & Embedding Engineer (Engineer 2)

### Objective
Create a vector embedding model for repositories using README content and metadata, and save them to the Vector Database.

### Status: 🔲 Not Started

### Input Data
Approved Repo Corpus (curated by Engineer 1).

### Key Tasks
1. **Text Embedding Pipeline**:
   - Set up a sentence transformer model (e.g., `all-MiniLM-L6-v2` or `BGE-small`).
   - Implement README chunking and paragraph text embedding.
2. **Categorical & Metadata Embeddings**:
   - Encode categorical values (primary languages, topics, star bands) into dense embeddings.
3. **Repo Mixer Model**:
   - Design and train/validate a projection layers model to combine README paragraph vectors with categorical metadata vectors into a final 384-dimensional Final Repo Embedding.
4. **Vector DB Indexing**:
   - Set up connection to Vector DB (e.g., `pgvector`).
   - Write scripts to write/upsert the Final Repo Embeddings to the vector index.

### Outputs & APIs
- API function: `generate_repo_embedding(repo_data: dict) -> np.ndarray`.
- Index collection populated in the Vector Database.

### Definition of Done (DoD)
- Unit tests verifying the output vector dimension is exactly 384.
- Validation check showing that repositories of similar topics (e.g., two React UI libraries) have a cosine similarity $> 0.75$.

---

## 🎟️ Ticket 3: User Representation & Persona Fusion Engineer (Engineer 3)

### Objective
Build a dynamic user profile system that represents user interests as real-time embeddings based on historical behavior and immediate session signals.

### Status: 🔲 Not Started

### Input Data
User interaction event streams (saves, stars, follows, clicks, skips, dwell time).
Final Repo Embeddings from Engineer 2.

### Key Tasks
1. **Long-term Persona Vector**:
   - Create an aggregation algorithm (e.g., attention-weighted average) of the embeddings of repositories the user has starred, followed, or liked historically.
2. **Short-term Persona Vector**:
   - Create a session-based embedding that averages recent clicks and saves.
   - Apply an exponential decay parameter ($\tau = 15$ mins) to fade out older session activities.
3. **Persona Fusion Gating**:
   - Write a fusion function to combine long-term and short-term persona embeddings (e.g., $E_{user} = \alpha E_{short} + (1 - \alpha) E_{long}$).
4. **Telemetry Vector Drift**:
   - Implement the mathematical equations that shift the short-term vector based on telemetry events (e.g., positive drift on saves/long dwell, negative drift on skips).

### Outputs & APIs
- API function: `update_user_embedding(user_id: str, event: dict) -> np.ndarray`.

### Definition of Done (DoD)
- Simulation tests showing the user embedding shifts towards "Machine Learning" when mock telemetry registers 5 consecutive clicks on Python ML repositories.

---

## 🎟️ Ticket 4: Multi-Source Candidate Retrieval Engineer (Engineer 4)

### Objective
Design a retrieval pipeline that screens millions of repos down to a candidate pool of under 1,000 using multiple retrieval strategies.

### Status: 🔲 Not Started

### Input Data
Final User Embedding (from Engineer 3).
Vector DB with Final Repo Embeddings (from Engineer 2).

### Key Tasks
1. **ANN Semantic Retrieval**: Implement a `pgvector`/vector index similarity search query using the Final User Embedding.
2. **Category & Trending Retrieval**: Query the Redis/Trending cache to extract top-ranking repos matching user interests.
3. **Exploration Retrieval**: Implement a multi-armed bandit or randomized adjacent-category retrieval module to fetch cold-start repositories.
4. **Freshness Retrieval**: Query recently pushed/active repositories.
5. **Merge & Deduplication**: Combine and deduplicate candidates from all 4 channels.

### Outputs & APIs
- API function: `retrieve_candidates(user_embedding: np.ndarray, user_interests: list) -> list[str]` (returns a list of candidate repo IDs, max length 1,000).

### Definition of Done (DoD)
- Latency bench validation verifying retrieval execution completes in under 30ms.
- Verification that candidates contain a mixture of semantic, trending, and exploratory items.

---

## 🎟️ Ticket 5: Multi-Stage Filtering & Diversity Engineer (Engineer 5)

### Objective
Narrow down the candidate pool of 1,000 repositories to a subset of 100 high-quality, diverse, and unique repositories.

### Status: 🔲 Not Started

### Input Data
Candidate pool of 1,000 repo IDs (from Engineer 4).
Quality scores (from Engineer 1) and Repo Embeddings (from Engineer 2).

### Key Tasks
1. **Hard Filters**: Filter out archived, dead, or low-quality repos (score below user-defined threshold).
2. **Forks & Duplicate Filter**: Run LSH or threshold cosine similarity checks to detect and drop repositories that are duplicates or forks.
3. **Diversity Engine**:
   - Implement a Maximal Marginal Relevance (MMR) algorithm to ensure topic and language diversity.
4. **Creative Diversity Injector**: Ingest interest-adjacent, high-trend repos to widen user discovery paths.

### Outputs & APIs
- API function: `filter_candidates(candidate_ids: list[str], user_profile: dict) -> list[str]` (returns a list of max 100 repo IDs).

### Definition of Done (DoD)
- Validation check verifying that the final 100 list contains no more than 3 repos from the same owner, and language distribution matches diversity thresholds.

---

## 🎟️ Ticket 6: Feature Engineering & Ranking Models Engineer (Engineer 6)

### Objective
Build pointwise classifiers to score user interest, click probability, and follow likelihood for the 100 candidates.

### Status: 🔲 Not Started

### Input Data
Filtered candidate list of 100 repo IDs (from Engineer 5).
User profiles and historical interaction logs.

### Key Tasks
1. **Ranking Feature Store**: Design feature vectors combining user metadata (interests, past languages) with repo metadata (primary language, topics, quality).
2. **Light Ranker**: Implement a GBDT model (e.g., LightGBM) to quickly score the 100 repositories.
3. **Heavy Pointwise Classifiers**:
   - **CTR Predictor**: Binary classification model predicting tap probability.
   - **Follow Predictor**: Classifier predicting whether the user will follow the repo/author.
   - **Dwell Predictor**: Regression model predicting dwell duration.

### Outputs & APIs
- API function: `score_candidates(user_id: str, filtered_ids: list[str]) -> dict[str, dict]` (returns predicted probabilities for click, follow, and dwell).

### Definition of Done (DoD)
- AUC score $> 0.72$ on offline click/follow validation datasets.

---

## 🎟️ Ticket 7: Sequence/GNN & Feed Assembly Engineer (Engineer 7)

### Objective
Apply graph neural networks and sequential session transitions to combine prediction scores, and format the final feed layout.

### Status: 🔲 Not Started

### Input Data
Pointwise prediction scores (from Engineer 6) and interaction graph.

### Key Tasks
1. **SASRec Sequential Recommender**: Implement a transformer model to capture sequence-based transitions of user interactions.
2. **Graph Neural Network (GNN)**: Set up a GNN model (e.g., GraphSAGE) over the user-repo-developer interaction graph to output network similarity scores.
3. **Score Fusion Engine**: Merge GNN scores, SASRec sequential scores, CTR predictions, and follow predictions using weighted linear regression.
4. **Feed Assembly & Post-Processing**:
   - *Freshness Injection*: Adjust score weights based on release/pushed date.
   - *Session layout constraint optimizer*: Prevent adjacent items from having identical languages or topics.

### Outputs & APIs
- API function: `generate_final_feed(user_id: str, candidates_scores: dict) -> list[str]` (returns an ordered list of 10-20 repo IDs).

### Definition of Done (DoD)
- Layout test verifying that no two items with the same owner or primary language are adjacent in the output feed.
