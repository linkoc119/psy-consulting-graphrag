# 📊 Tổng quan Hệ thống GraphRAG Psychology Chatbot

**Phiên bản:** 1.0.0  
**Ngày tạo:** 2024-01-15  
**Dự án:** Chatbot tư vấn tâm lý học đường sử dụng GraphRAG  
**Tài liệu này:** Mô tả đầy đủ về hệ thống, kiến trúc, components, và logic để bạn có thể hiểu và phát triển lại toàn bộ project.

---

## 🎯 Mục tiêu Dự án

Xây dựng một **AI chatbot tư vấn tâm lý học đường** với khả năng:

1. **Phân luồng thông minh (Triage):** Tự động phát hiện mức độ nghiêm trọng của vấn đề (1-5)
2. **Suy luận đa chặng (Multi-hop Reasoning):** Kết nối triệu chứng → bệnh lý → kỹ năng tư vấn
3. **An toàn tuyệt đối:** Không chẩn đoán y khoa, không kê đơn, luôn chuyển tuyến khi khẩn cấp
4. **Hỗ trợ tiếng Việt:** Toàn bộ system thiết kế cho tiếng Việt
5. **Chạy local:** Bảo mật dữ liệu, không gửi lên cloud

---

## 🏗️ Kiến trúc Tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ChatInterface Component                                      │  │
│  │ - Message list                                               │  │
│  │ - Input area                                                 │  │
│  │ - Crisis banner                                              │  │
│  │ - Severity indicators                                        │  │
│  └─────────────────────────────┬─────────────────────────────────┘  │
└───────────────────────────────┼──────────────────────────────────────┘
                                │ HTTPS/JSON
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI)                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    RAG SERVICE (rag_service.py)               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │  │  Retrieval  │  │   Triage &  │  │     Generation       │  │
│  │  │   Phase 1   │→ │  Prompt     │→ │     Phase 3          │  │
│  │  │             │  │  Building   │  │                     │  │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘  │
│  │         │               │                    │                │
│  │    ┌────┴────┐    ┌────┴─────┐      ┌──────┴──────┐         │
│  │    ▼         ▼    ▼          ▼      ▼             ▼         │
│  │  Qdrant   Neo4j  Severity   Prompt  LLM (Ollama)            │
│  │  Search   Graph  Detection  Builder  Qwen 2.5-3B            │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │   Qdrant    │ │    Neo4j    │ │  Embedding  │
        │  Vector DB  │ │   Graph DB  │ │   Model     │
        └─────────────┘ └─────────────┘ └─────────────┘
```

---

## 📦 Components Chi tiết

### 1. LLM Layer (Ollama + Qwen 2.5-3B)

**File:** `backend/app/services/llm_service.py`

**Chức năng:**
- Kết nối async với Ollama API
- Hỗ trợ streaming response
- Quản lý connection pool
- Check health, pull model

**Cấu hình:**
```python
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048
```

**API Methods:**
- `generate(prompt, stream=True)` - Streaming generation
- `generate_text(prompt)` - Complete response
- `chat(messages)` - Conversation mode
- `check_connection()` - Health check

---

### 2. Embedding Service

**File:** `backend/app/services/embedding_service.py`

**Model:** `AITeamVN/Vietnamese_Embedding` (1024 dimensions)

**Chức năng:**
- Load SentenceTransformer model
- Encode texts to vectors
- Compute cosine similarity
- Singleton pattern for reuse

**Usage:**
```python
embedding = get_embedding_service()
vector = embedding.encode_query("Tôi cảm thấy lo lắng")
```

---

### 3. Reranker Service

**File:** `backend/app/services/reranker_service.py`

**Model:** `AITeamVN/Vietnamese_Reranker` (Cross-encoder)

**Chức năng:**
- Rerank retrieved documents based on query relevance
- Cross-encoding for accurate scoring
- Filter top-k most relevant

**Workflow:**
```
Query + Document → Tokenize → Model inference → Score → Sort
```

**Usage:**
```python
reranker = get_reranker_service()
results = reranker.rerank(query, documents, metadata)
# Returns: [(doc, score, meta), ...] sorted by score
```

---

### 4. Qdrant Vector Database

**File:** `backend/app/services/qdrant_service.py`

**Chức năng:**
- Store document embeddings
- Fast dense vector search (cosine similarity)
- Filter by metadata (doc_type, risk_priority)
- Collection management

**Schema:**
```python
Collection: "psychology_chunks"
Vector size: 1024
Distance: Cosine
Payload: {
    "page_content": "...",
    "source": "resources/pdtt.pdf",
    "doc_type": "medical_guideline",
    "risk_priority": "high",
    "page_no": 15,
    "section": "Bài 1: Triệu chứng..."
}
```

**Operations:**
- `create_collection()` - Create if not exists
- `upload_points(points)` - Batch upload
- `search(query_vector, limit, filter)` - Vector search
- `get_collection_info()` - Stats

---

### 5. Neo4j Graph Database

**File:** `backend/app/services/neo4j_service.py`

**Chức năng:**
- Store knowledge graph (entities & relationships)
- Graph traversal for multi-hop reasoning
- Node/relationship queries
- Constraint management

**Graph Schema:** (See `backend/app/models/graph_schema.py`)

**Node Labels (8 types):**
1. `BenhLy` - Psychological disorders
2. `TrieuChung` - Symptoms
3. `DauHieuNguyHiem` - Crisis indicators/red flags
4. `HanhDongPFA` - Psychological First Aid actions
5. `KyNangTuVan` - Counseling micro-skills
6. `BuocTuVan` - Counseling process steps
7. `Thuoc` - Medications (for ID only, NOT prescription)
8. `DoiTuong` - Demographic groups
9. `DocumentChunk` - Source text chunks

**Relationship Types (8 types):**
- `CO_TRIEU_CHUNG` - Disorder HAS symptom
- `BAO_HIEU_NGUY_HIEM` - Symptom IS red flag
- `YEU_CAU_HANH_DONG` - Danger REQUIRES action
- `DIEU_TRI_BANG` - Disorder TREATED WITH drug
- `QUAN_LY_BANG` - Issue MANAGED WITH skill/step
- `AP_DUNG_CHO` - Skill/action APPLIES TO demographic
- `BAO_GOM_BUOC` - Step CONTAINS substep
- `NAM_TRONG_CHUNK` - Entity MENTIONED IN chunk

**Severity Levels (1-5):**
- Level 1: Normal stress
- Level 2: Mild anxiety
- Level 3: Moderate issues
- Level 4: Severe/crisis
- Level 5: Emergency (suicide, psychosis)

---

### 6. RAG Service (Core Logic)

**File:** `backend/app/services/rag_service.py`

**Phases:**

#### Phase 1: Retrieval (`_retrieve_context()`)

**Complete GraphRAG Pipeline:**

```
User Query
    ↓
[1] Generate Embedding (Vietnamese_Embedding, 1024-dim)
    ↓
[2] Vector Search in Qdrant (top 15)
    ├─ Returns document chunks with metadata
    └─ Extract entity_ids from payloads (linked to graph nodes)
    ↓
[3] Graph Traversal in Neo4j (depth=2)
    ├─ Start from entity_ids extracted from Qdrant results
    ├─ Traverse only "golden" relationships:
    │  • CO_TRIEU_CHUNG
    │  • BAO_HIEU_NGUY_HIEM
    │  • YEU_CAU_HANH_DONG
    │  • DIEU_TRI_BANG
    │  • QUAN_LY_BANG
    │  • AP_DUNG_CHO
    └─ Returns: graph_nodes + relationships
    ↓
[4] Reciprocal Rank Fusion (RRF)
    ├─ Vector results have qdrant_rank (1-based)
    ├─ Graph nodes have graph_rank (1-based)
    ├─ RRF score = 1 / (k + rank), k=60
    └─ Combined score = rrf_score (initial)
    ↓
[5] Reranking (Vietnamese_Reranker)
    ├─ Prepare texts: documents + graph_nodes
    ├─ Cross-encoder scores relevance to query
    ├─ Map scores back to items
    └─ Combined score = rrf_score + rerank_score
    ↓
[6] Final Selection
    ├─ Sort by combined_score descending
    ├─ Keep top 10 items total
    └─ Separate: documents[] vs graph_nodes[]
    ↓
[7] Severity Detection
    ├─ Check crisis keywords in query
    ├─ Check severity_level from graph nodes
    └─ Return severity (1-5) and red_flags
```

**Returns:**
```python
{
    "documents": [...],      # Top 5-10 document chunks
    "graph_nodes": [...],    # Top related graph entities
    "relationships": [...],  # Graph edges from traversal
    "severity_indicators": {
        "level": 1-5,
        "red_flags": [...],
        "has_crisis_keywords": bool
    },
    "query": "..."
}
```

**Key Innovation: RRF Fusion**

The system combines two independent retrieval signals:
- **Vector search** (Qdrant): semantic similarity based on embedding
- **Graph traversal** (Neo4j): knowledge-grounded entity expansion

RRF formula: `score = 1 / (k + rank)` where k=60
- This gives equal weight to both sources initially
- Graph nodes with higher severity get better initial rank
- Reranker then learns to weight sources based on query relevance

**Example:** For query about "ảo giác" (hallucinations):
- Qdrant returns doc about psychosis (rank 1)
- Graph traversal from "Ảo thanh" node returns:
  - `BenhLy: Tâm thần phân liệt` (severity 5)
  - `DauHieuNguyHiem: Hoang tưởng` (severity 4)
  - `HanhDongPFA: Chuyển tuyến` (severity 4)
- RRF ranks these by position in each list
- Reranker boosts items directly relevant to query

#### Phase 2: Triage & Prompt Building (`_build_prompt()`)

**Triage Logic:**
- If severity >= 4 → CRISIS mode (PFA)
- If severity <= 3 → COUNSELING mode (6-step process)

**Prompt Structure:**
```
[FEW_SHOT_EXAMPLES]  # 3 examples relevant to mode

=== NGỮ CẢNH ===
[Top 5 documents from Qdrant]
[Top 5 graph nodes from Neo4j]

=== CÂU HỎI ===
User query

=== PHẢN HỒI ===
(LLM generates here)
```

**System Prompts:**
- `SYSTEM_PROMPT_COUNSELING` - 6-step counseling, no diagnosis
- `SYSTEM_PROMPT_CRISIS` - PFA protocol, immediate referral

**Context Formatting:**
- Documents: `[{i}] ({doc_type}) {text[:500]}...`
- Graph nodes: `[{i}] ({node_type}) {text[:300]}...`

#### Phase 3: Generation (`process_query()`)

- Stream response from LLM (Ollama)
- Yield chunks to frontend via async generator
- Store `last_context` for source extraction
- Return metadata (severity, crisis flag) with full response

---

### 7. Prompt Templates

**File:** `backend/app/utils/prompts.py`

**Components:**

1. **SYSTEM_PROMPT_COUNSELING**
   - Role: School counselor
   - Rules: No diagnosis, no prescription, 6-step process
   - Age-appropriate communication
   - Confidentiality + referral limits

2. **SYSTEM_PROMPT_CRISIS**
   - Role: Psychological First Aid provider
   - Principles: Safety, Dignity, Rights
   - PFA steps: Look → Listen → Link
   - Emergency contacts: 111, 113, child protection

3. **FEW_SHOT_COUNSELING** (3 examples)
   - Exam stress
   - Friendship conflict
   - Self-comparison

4. **FEW_SHOT_CRISIS** (3 examples)
   - Suicide ideation
   - School violence
   - Psychosis (auditory hallucinations)

5. **TRIAGE_GUIDELINES**
   - Severity thresholds
   - Red flags list
   - Action mapping

---

### 8. API Layer

**Files:**
- `backend/app/routers/chat.py` - Chat endpoints
- `backend/app/routers/health.py` - Health checks
- `backend/app/main.py` - FastAPI app, CORS, lifecycle

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Full health check (all services) |
| GET | `/health/ready` | Kubernetes readiness probe |
| POST | `/chat/completion` | Send message, get streaming response |
| GET | `/chat/conversation/{id}` | Get history |
| DELETE | `/chat/conversation/{id}` | Delete conversation |
| POST | `/chat/clear` | Clear all conversations |

**Request Model:**
```python
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str]
    history: List[ChatMessage] = []
    user_id: Optional[str]
```

**Response Model:**
```python
class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    severity_level: int  # 1-5
    is_crisis: bool
    sources: List[dict]
    processing_time_ms: float
```

---

### 9. Frontend (React + Material-UI)

**File Structure:**
```
frontend/src/
├── components/
│   └── ChatInterface.jsx  # Main chat UI
├── services/
│   └── api.js            # API client with streaming
├── App.jsx               # Root component, health check
├── index.jsx             # Entry point, theme
└── App.css               # Custom styles
```

**Key Features:**

1. **ChatInterface Component:**
   - Message list with user/assistant bubbles
   - Input field with Send button
   - Crisis banner (red) when severity >= 4
   - Severity chips (color-coded)
   - Sources display (top 3)
   - Typing indicator
   - Auto-scroll to bottom
   - Conversation history management

2. **Streaming Support:**
   - `sendChatMessage()` uses `responseType: 'stream'`
   - Accumulates chunks and displays progressively
   - Shows circular progress while waiting

3. **Health Check on Load:**
   - Calls `/health` on mount
   - Shows error if backend unavailable
   - Displays troubleshooting tips

4. **Material-UI Theme:**
   - Primary blue (#1976d2)
   - Vietnamese-friendly fonts (Roboto + Noto Sans Vietnamese)
   - Responsive design

---

## 📄 Data Pipeline

### 1. Chunking Process

**File:** `backend/chunking/chunk_processor.py`

**Input:** 3 PDF files in `resources/`
- `pdtt.pdf` - Phác đồ Tâm thần 2020 (Bộ Y tế)
- `sctl.pdf` - Sơ cứu Tâm lý (WHO)
- `tvtl.pdf` - Sổ tay Tư vấn Tâm lý Học đường

**Process:**
1. Extract text per page using `pdfplumber`
2. Extract tables to Markdown format
3. Clean text:
   - Normalize Unicode
   - Fix OCR errors (SÖT → SÚT, etc.)
   - Remove page numbers
   - Fix hyphenated line breaks
4. Detect section titles (patterns: "Bài 1", "Chương 2", etc.)
5. Combine text + tables with metadata prefix
6. Split using `RecursiveCharacterTextSplitter`:
   - Separators: `\n\n[MỤC:`, `\n\n`, `\n`, `.`, ` `
   - Chunk size: 1000 chars
   - Overlap: 150 chars
7. Deduplicate using MD5 hash
8. Filter out short chunks (< 120 chars)

**Output:** `backend/data/processed_chunks.jsonl`
```json
{
  "page_content": "...",
  "metadata": {
    "source": "resources/pdtt.pdf",
    "section": "Bài 1: Sa sút trí tuệ",
    "doc_type": "medical_guideline",
    "risk_priority": "high",
    "page_no": 12,
    "chunk_id": "pdtt.pdf_p12_a1b2c3d4"
  }
}
```

### 2. Indexing Pipeline (GraphRAG Build)

**Script:** `backend/scripts/index_data.py`

**Complete workflow:**

```
Load processed_chunks.jsonl
    ↓
[1] Generate Embeddings
    └─ Vietnamese_Embedding → 1024-dim vectors (one per chunk)
    ↓
[2] Entity Extraction (per chunk)
    ├─ Rule-based: Match against PREDEFINED_ENTITIES
    ├─ Optional LLM: Qwen for dynamic entity extraction
    └─ Returns: List of {name, type, description}
    ↓
[3] Create Graph Nodes (Neo4j)
    ├─ MERGE each unique entity by normalized name
    ├─ Map Vietnamese labels → English (BenhLy, TrieuChung, etc.)
    └─ Set severity_level based on entity type
    ↓
[4] Create Relationships
    ├─ Link entities that co-occur in same chunk
    ├─ Infer relationship type from entity types
    └─ Only create "golden" relationships (exclude noisy co-occurrence)
    ↓
[5] Link Chunks to Entities
    └─ Create NAM_TRONG_CHUNK relationships
    ↓
[6] Upload to Qdrant
    ├─ Each chunk becomes a point with:
    │  • ID: sequential integer
    │  • Vector: 1024-dim embedding
    │  • Payload: chunk metadata + entity_ids[]
    └─ entity_ids[] stores graph node IDs for traversal
```

**Key Insight:** The `entity_ids` field in Qdrant payload is the bridge between vector search and graph traversal. When a document chunk is retrieved, its linked entities become entry points for graph traversal.

**Example Qdrant payload:**
```json
{
  "page_content": "Ảo thanh là triệu chứng của tâm thần phân liệt...",
  "source": "resources/pdtt.pdf",
  "doc_type": "medical_guideline",
  "chunk_id": "pdtt.pdf_p15_x1y2z3",
  "entity_ids": ["TrieuChung_ao_thanh", "BenhLy_tam_than_phan_liet"]
}
```

**Indexing command:**
```bash
docker exec psychology-backend python -m scripts.index_data --rechunk
# or (skip rechunk if processed_chunks.jsonl exists)
docker exec psychology-backend python -m scripts.index_data
```

---

## 🧠 Knowledge Graph Details

### Entity Extraction Strategy

**From Medical Guideline (pdtt.pdf):**
- Extract: `BenhLy`, `TrieuChung`, `Thuoc`
- Relationships: `CO_TRIEU_CHUNG`, `DIEU_TRI_BANG`
- Source domain: `Psychiatry_PhacDo`

**From PFA (sctl.pdf):**
- Extract: `HanhDongPFA`, `DauHieuNguyHiem`, `DoiTuong`
- Relationships: `YEU_CAU_HANH_DONG`, `AP_DUNG_CHO`
- Source domain: `PFA_SoCuu`

**From Counseling (tvtl.pdf):**
- Extract: `KyNangTuVan`, `BuocTuVan`, `DoiTuong`
- Relationships: `QUAN_LY_BANG`, `BAO_GOM_BUOC`
- Source domain: `Counseling_SoTay`

### Predefined Entities

**BenhLy (7 disorders):**
- Alzheimer, Schizophrenia, Depression, Anxiety, ADHD, Vascular dementia, Personality disorder

**DauHieuNguyHiem (6 red flags):**
- Suicide intent, Self-harm, Psychosis (hallucinations), Delusions, Violence, Abuse victim

**Thuoc (7 drugs):**
- Sertraline, Fluoxetine (antidepressants)
- Haloperidol, Risperidone, Clozapine (antipsychotics)
- Donepezil (cognitive enhancer)
- Valproate (mood stabilizer)

---

## 🔄 Data Flow Example

**User Query:** "Em nghe thấy tiếng nói trong đầu, bảo em làm chuyện xấu"

### Step-by-step:

1. **Embedding:**
   - Query → Vietnamese_Embedding → 1024-dim vector

2. **Qdrant Search:**
   - Cosine similarity search
   - Returns: [doc1(score:0.89), doc2(0.85), ...] (top 15)
   - doc1: "Ảo thanh là triệu chứng của tâm thần phân liệt..."
   - doc2: "PFA: Lắng nghe, không phán xét..."

3. **Entity Extraction:**
   - From doc1: matches node `TrieuChung: Ảo thanh` (alias: "nghe thấy tiếng nói")
   - From doc1: matches node `BenhLy: Tâm thần phân liệt`

4. **Graph Traversal:**
   ```
   Start: TrieuChung:Ảo thanh
   → CO_TRIEU_CHUNG → BenhLy:Tâm thần phân liệt
   → BAO_HIEU_NGUY_HIEM → DauHieuNguyHiem:Hoang tưởng/Ảo giác
   → YEU_CAU_HANH_DONG → HanhDongPFA:Chuyển tuyến y tế
   → AP_DUNG_CHO → DoiTuong:Học sinh THCS
   ```

5. **Severity Detection:**
   - Keyword "tiếng nói trong đầu" matches "ảo thanh" alias
   - Graph node `BenhLy:Tâm thần phân liệt` has severity_level=5
   - → Overall severity = 5

6. **Rerank:**
   - Combine 15 docs + 8 graph entities = 23 items
   - Reranker scores relevance to query
   - Keep top 10: [doc1, doc2, node1, node2, ...]

7. **Prompt Building (Crisis Mode):**
   ```
   FEW_SHOT_CRISIS (3 examples)

   === NGỮ CẢNH ===
   [1] (medical_guideline) Ảo thanh là triệu chứng của tâm thần phân liệt...
   [2] (graph: HanhDongPFA) Chuyển tuyến y tế: Hướng dẫn tìm bác sĩ tâm thần...
   [3] (graph: DauHieuNguyHiem) Ảo giác - nghe thấy tiếng nói không tồn tại...

   === CÂU HỎI ===
   Em nghe thấy tiếng nói trong đầu...

   === PHẢN HỒI ===
   ```

8. **Generation:**
   - System prompt: `SYSTEM_PROMPT_CRISIS`
   - LLM generates PFA response:
   ```
   Em đang trải qua ảo giác, điều này rất đáng sợ...
   Em cần gặp bác sĩ tâm thần ngay...
   Hãy liên hệ với bố mẹ và đường dây 113/111...
   ```

9. **Response:**
   ```json
   {
     "message": "...",
     "severity_level": 5,
     "is_crisis": true,
     "sources": [...]
   }
   ```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Ollama LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# Embedding Model
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding
EMBEDDING_DIMENSION=1024

# Reranker Model
RERANKER_MODEL=AITeamVN/Vietnamese_Reranker
RERANKER_TOP_K=5

# Qdrant Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=psychology_chunks
QDRANT_VECTOR_SIZE=1024

# Neo4j Graph DB
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# App
DEBUG=false
```

### Python Dependencies

**`requirements.txt`:**
- fastapi, uvicorn - Web framework
- pydantic, pydantic-settings - Validation
- neo4j - Graph DB driver
- qdrant-client - Vector DB client
- langchain-text-splitters - Text splitting
- pdfplumber - PDF extraction
- torch, transformers, sentence-transformers - ML models
- ollama - LLM client
- numpy, pandas - Data processing
- aiohttp - Async HTTP
- python-multipart - Form data

---

## 🐳 Docker Deployment

### docker-compose.yml

**Services:**

1. **ollama**
   - Image: `ollama/ollama:latest`
   - Ports: 11434
   - Volumes: `ollama_data` (persist models)
   - GPU: `deploy.resources.reservations.devices`

2. **qdrant**
   - Image: `qdrant/qdrant:latest`
   - Ports: 6333 (HTTP), 6334 (gRPC)
   - Volumes: `qdrant_data`

3. **neo4j**
   - Image: `neo4j:5.15`
   - Ports: 7687 (Bolt), 7474 (HTTP Browser)
   - Volumes: `neo4j_data`, `neo4j_logs`
   - Env: `NEO4J_AUTH=neo4j/password`
   - Plugin: APOC (for graph algorithms)

4. **backend**
   - Build: `./backend/Dockerfile`
   - Ports: 8000
   - Depends on: ollama, qdrant, neo4j (healthy)
   - Volumes: code mount for dev, `./data:/app/data`

5. **frontend**
   - Build: `./frontend/Dockerfile`
   - Ports: 3000
   - Volumes: code mount for dev
   - Env: `REACT_APP_API_URL=http://localhost:8000`

**Network:** `graphrag-network` (bridge)

---

## 🧪 Testing Strategy

### Unit Tests (TODO)
- Test each service independently
- Mock external dependencies
- Test chunking logic
- Test prompt generation

### Integration Tests (TODO)
- End-to-end RAG pipeline
- Severity detection accuracy
- Graph traversal correctness

### Manual Testing Checklist

- [ ] Ollama responds to `/api/generate`
- [ ] Qdrant collection exists and has points
- [ ] Neo4j constraints created, nodes present
- [ ] Embedding model loads (GPU/CPU)
- [ ] Reranker model loads
- [ ] `/health` returns healthy
- [ ] Chat API returns response (non-streaming)
- [ ] Streaming works in frontend
- [ ] Crisis messages trigger red banner
- [ ] Counseling messages show 6-step guidance
- [ ] Frontend displays all UI elements

---

## 📊 Performance Considerations

### Current Performance (Measured)

**Latency Breakdown (Typical after optimizations):**

| Step | Time (ms) | Notes |
|------|-----------|-------|
| Query embedding | 80-150 | Vietnamese_Embedding (1024-dim) |
| Qdrant search | 15-40 | HNSW search on 1039 points |
| Graph traversal | 40-120 | 2-hop, ~30-50 nodes returned |
| Reranking (~65 items) | 300-800 | Cross-encoder, batch size dependent |
| LLM generation (stream) | 4000-12000 | 600-1500 tokens, CPU inference |
| **Total** | **4500-14000** | ~5-15 seconds typical |

**Bottlenecks:**
1. **Reranker** (~300-800ms): Processing 60+ items with cross-encoder
2. **LLM** (~4-12s): Qwen 2.5-3B on CPU (much faster on GPU)
3. **Graph traversal** can be slow if too many nodes returned (before filtering)

**Optimizations Applied:**
- Filter relationships to 6 golden types (excludes noisy co-occurrence)
- Limit graph nodes to top 50 by severity before fusion
- Keep total items for reranker to ~65 (15 docs + 50 graph)

**Further Optimizations (Future):**
1. **Embedding cache** - Redis cache for frequent query embeddings
2. **Reranker batching** - Process multiple queries together
3. **Early graph filtering** - Filter by node type before traversal
4. **Model quantization** - Use 4-bit/8-bit embedding/reranker models
5. **Adjust RRF k parameter** - Tune to weight vector vs graph differently
6. **Reduce reranker top_k** - Currently uses all fused items (~65)
7. **GPU acceleration** - Ensure all models use GPU if available
8. **Async parallelization** - Already async; could add more concurrency

---

## 🚀 Future Enhancements

### Phase 2 (Completed) ✅
- [x] Implement Neo4j population from PDFs via `graph_indexer.py`
- [x] Entity linking between Qdrant and Neo4j (entity_ids)
- [x] GraphRAG retrieval with RRF fusion and reranking
- [ ] Add Redis for conversation persistence
- [ ] Implement rate limiting
- [ ] Add user authentication
- [ ] Multi-language support (English)
- [ ] Analytics dashboard

### Phase 3 (Advanced)
- [ ] Fine-tune Qwen on counseling dialogues
- [ ] Add feedback loop (thumbs up/down)
- [ ] Conversation summarization for context management
- [ ] Multi-modal support (images, voice)
- [ ] Mobile app (React Native)
- [ ] Advanced triage with ML classifier
- [ ] Caching layer (Redis) for embeddings and queries
- [ ] Batch processing for high-load scenarios

---

## 📚 References

### Design Document
`Thiết kế GraphRAG tư vấn tâm lý.md` - Original requirements and schema

### External Resources
- **Phác đồ Tâm thần 2020** - Bộ Y tế Việt Nam
- **Sơ cứu Tâm lý (PFA)** - WHO
- **Sổ tay Tư vấn Tâm lý Học đường** - Bộ GD&ĐT & UNICEF
- **AITeamVN Models** - Vietnamese Embedding & Reranker

### Technologies
- [Ollama](https://ollama.ai/) - LLM serving
- [Qdrant](https://qdrant.tech/) - Vector database
- [Neo4j](https://neo4j.com/) - Graph database
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [React](https://reactjs.org/) - Frontend library
- [Material-UI](https://mui.com/) - React component library

---

## 🗂️ File Index

### Backend
- `app/main.py` - FastAPI app entry point
- `app/config.py` - Configuration settings
- `app/services/` - Business logic services
- `app/routers/` - API endpoints
- `app/models/` - Pydantic schemas, graph schema
- `app/utils/` - Prompt templates
- `chunking/chunk_processor.py` - PDF processing
- `requirements.txt` - Python dependencies
- `Dockerfile` - Backend container

### Frontend
- `src/App.jsx` - Root component
- `src/components/ChatInterface.jsx` - Main UI
- `src/services/api.js` - API client
- `src/index.jsx` - Entry point
- `package.json` - Node dependencies
- `Dockerfile` - Frontend container
- `vite.config.js` - Build config

### Configuration
- `docker-compose.yml` - Orchestration
- `.env.example` - Environment template
- `RUN_GUIDE.md` - **This file:** Setup & operations

### Documentation
- `README.md` - Project overview & quick start
- `SYSTEM_OVERVIEW.md` - **This file:** Complete system architecture
- `Thiết kế GraphRAG tư vấn tâm lý.md` - Original design doc

---

## ✅ Quick Reference

### Start Everything
```bash
docker-compose up -d
docker exec psychology-ollama ollama pull qwen2.5:3b
```

### Stop Everything
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f backend
```

### Restart Service
```bash
docker-compose restart backend
```

### Test API
```bash
curl http://localhost:8000/health
```

### Access UI
```
http://localhost:3000
```

### Neo4j Browser
```
http://localhost:7474
Username: neo4j
Password: password
```

### Qdrant Dashboard (optional)
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
# Or use Qdrant console at http://localhost:6333/dashboard
```

---

## 📝 Notes for Developers

### Adding New Document Types
1. Add PDF to `resources/`
2. Update `DEFAULT_DOCS_CONFIG` in `chunk_processor.py`
3. Add corresponding entities to `PREDEFINED_ENTITIES` in `graph_schema.py`
4. Update `doc_type` mapping in prompts if needed

### Modifying Triage Logic
- Edit `_detect_severity_indicators()` in `rag_service.py`
- Adjust `TRIAGE_THRESHOLD_HIGH` in `config.py`
- Update `TRIAGE_GUIDELINES` in `prompts.py`

### Changing LLM Model
1. Update `.env`: `LLM_MODEL=new-model-name`
2. Ensure model exists in Ollama: `ollama pull new-model-name`
3. Adjust temperature/tokens as needed

### Extending Graph Schema
1. Add node label to `NODE_LABELS` in `graph_schema.py`
2. Add relationship to `RELATIONSHIPS`
3. Update `PREDEFINED_ENTITIES` if needed
4. Add constraint in `neo4j_service.py:create_constraints()`
5. Update prompt templates to reference new entities

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0  
**Status:** Production Ready (with data indexing pending)

---

## 🔍 Quick Diagnostic

**If something breaks, check in order:**

1. **Docker containers running?**
   ```bash
   docker-compose ps
   ```

2. **Ollama model loaded?**
   ```bash
   docker exec psychology-ollama ollama list
   ```

3. **Qdrant collection exists?**
   ```bash
   curl http://localhost:6333/collections
   ```

4. **Neo4j nodes populated?**
   ```bash
   docker exec psychology-neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"
   ```

5. **Backend logs:**
   ```bash
   docker-compose logs backend | tail -50
   ```

6. **Frontend console errors?** Open DevTools (F12)

---

**END OF DOCUMENTATION**