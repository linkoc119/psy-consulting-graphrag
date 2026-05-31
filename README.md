# GraphRAG Psychology Chatbot

Hệ thống chatbot tư vấn tâm lý học đường sử dụng kiến trúc GraphRAG kết hợp Vector Database (Qdrant) và Graph Database (Neo4j).

## Tổng quan

Đây là một hệ thống AI hỗ trợ tư vấn tâm lý cho học sinh, với khả năng:

- **Phân luồng khẩn cấp (Triage)**: Tự động phát hiện các dấu hiệu nguy hiểm và chuyển sang chế độ sơ cứu
- **Query Classification**: Phân loại tự động crisis/counseling/general với conversation history
- **Doc-Type Filtering**: Counseling queries chỉ retrieve school_counseling docs
- **Suy luận đồ thị (Graph Reasoning)**: Kết nối triệu chứng - bệnh lý - kỹ năng tư vấn
- **Tìm kiếm ngữ nghĩa (Vector Search)**: Truy xuất tài liệu liên quan
- **Reranking thông minh**: Lọc và ưu tiên thông tin quan trọng nhất
- **Streaming response**: Phản hồi theo thời gian thực
- **Configurable LLM provider**: Có thể chọn Ollama local, OpenAI-compatible API hoặc Claude-compatible provider qua `backend/.env`

### GraphRAG Pipeline

```
Query → Embedding (1024-dim) → Query Classification
                                ↓
              (counseling → filter doc_type=school_counseling)
                                ↓
          Qdrant search (top 5) → Extract entity_ids
                                ↓
         Neo4j traversal (max 3 entities, depth=1)
                                ↓
                  RRF Fusion (k=60) + Limit to 10
                                ↓
                        Reranking (top 3)
                                ↓
                 LLM Generation (configured provider)
```

## Kiến trúc

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI     │────│   FastAPI      │────│ LLM Provider   │
│   (Port 3000)   │    │   (Port 8000)  │    │ Ollama/API     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │   Qdrant     │ │    Neo4j     │ │  Embedding   │
           │  Vector DB   │ │   Graph DB   │ │   Model      │
           └──────────────┘ └──────────────┘ └──────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
           ┌─────────────────┐ ┌─────────────────┐
           │  Reranker       │ │   Graph        │
           │  (Cross-encoder)│ │  Traversal     │
           └─────────────────┘ └─────────────────┘
```

## Công nghệ sử dụng

| Thành phần | Công nghệ | Mô tả |
|------------|-----------|-------|
| **LLM** | Ollama / OpenAI-compatible / Claude-compatible | Cấu hình bằng `LLM_PROVIDER` trong `backend/.env` |
| **Vector DB** | Qdrant | Lưu trữ embeddings 1024-dim và tìm kiếm cosine |
| **Graph DB** | Neo4j | Lưu trữ knowledge graph với APOC |
| **Embedding** | AITeamVN/Vietnamese_Embedding | Mô hình nhúng tiếng Việt (1024-dim) |
| **Reranker** | AITeamVN/Vietnamese_Reranker | Reranking cross-encoder (GPU auto-detect) |
| **Backend** | FastAPI (Python) | REST API với async support |
| **Frontend** | React + Material-UI | Giao diện người dùng |

## Cài đặt nhanh

### Yêu cầu

- Docker & Docker Compose
- NVIDIA GPU (khuyến nghị cho Ollama) hoặc CPU
- 16GB+ RAM (8GB tối thiểu)

### Bước 1: Clone và cấu hình

```bash
git clone <your-repo>
cd psy-consulting-graphrag
# Backend reads configuration from backend/.env.
# Chỉnh sửa backend/.env nếu cần (mặc định hợp lệ cho local)
```

### Bước 2: Khởi động infrastructure services

```bash
# CPU/default
docker compose up -d ollama qdrant neo4j
```

Mặc định `docker-compose.yml` chạy bằng CPU, không yêu cầu NVIDIA runtime. Nếu máy có NVIDIA GPU và Docker đã hỗ trợ GPU, có 2 lựa chọn override:

```bash
# GPU full: backend retrieval/reranker + Ollama đều có GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama qdrant neo4j

# Hybrid: retrieval/backend CPU, Ollama LLM GPU
docker compose -f docker-compose.yml -f docker-compose.hybrid.yml up -d ollama qdrant neo4j
```

Khi vừa sửa cấu hình Docker Compose hoặc chuyển giữa CPU/GPU/Hybrid mode, nên recreate container:

```bash
# CPU/default
docker compose up -d --build --force-recreate

# GPU full
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --force-recreate

# Hybrid: retrieval CPU, LLM GPU
docker compose -f docker-compose.yml -f docker-compose.hybrid.yml up -d --build --force-recreate
```

Sau bước 4, các dịch vụ sẽ được khởi động:
- Ollama (port 11434)
- Qdrant (port 6333)
- Neo4j (ports 7687, 7474)
- Backend API (port 8000)
- Frontend (port 3000)

### Bước 3: Pull model Qwen

```bash
# Pull model Qwen vào volume ollama_data
docker exec psychology-ollama ollama pull qwen2.5:3b-instruct-q6_K

# Check downloaded model
docker exec psychology-ollama ollama list
```

### Bước 4: Khởi động backend và frontend

```bash
# CPU/default
docker compose up -d --build

# GPU
# docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Hybrid: retrieval CPU, LLM GPU
# docker compose -f docker-compose.yml -f docker-compose.hybrid.yml up -d --build
```

**Lần đầu tiên:** Backend sẽ pre-load ML models (embedding + reranker) trong startup → mất **5-10 phút**.
Kiểm tra logs:
```bash
docker compose logs backend -f
# Đợi dòng: "✅ All services initialized and models loaded - ready for instant response!"
```

### Bước 5: Kiểm tra health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

`/health/ready` là readiness check nhẹ dùng cho Docker healthcheck. `/health` là deep health check cho LLM/Qdrant/Neo4j.

### Bước 6: Mở browser

Truy cập: http://localhost:3000

## Chunking & Indexing

```python
# backend/config.py
EMBEDDING_DIMENSION = 1024
QDRANT_VECTOR_SIZE = 1024
```

### 1. Process PDFs into Chunks

**Using Docker (Recommended):**

```bash
# Run chunking inside the backend container
docker exec psychology-backend python -m chunking.chunk_processor
```

### 2. Index into Qdrant & Neo4j (GraphRAG)

**Indexing Commands:**

```bash
# Index with rule-based entity extraction (fast)
docker exec psychology-backend python -m scripts.index_data

# With LLM-based extraction (more accurate, slower)
docker exec psychology-backend python -m scripts.index_data --use-llm

# ⚠️ Lưu ý: LLM-based extraction có thể fallback về rule-based nếu LLM không trả về JSON hợp lệ.
# Kiểm tra backend logs để biết chi tiết.

# Re-chunk and re-index
docker exec psychology-backend python -m scripts.index_data --rechunk

# Clear everything and re-index
docker exec psychology-backend python -m scripts.index_data --clear-db --rechunk
```

### 3. Verify Graph & Vector Database

```bash
# Check Qdrant collection info
curl http://localhost:6333/collections/psychology_chunks | jq

# Check entity_ids in a point
docker exec psychology-backend python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://qdrant:6333')
result = client.scroll(collection_name='psychology_chunks', limit=1, with_payload=True)
if result[0]:
    point = result[0][0]
    print('entity_ids count:', len(point.payload.get('entity_ids', [])))
"

# Check Neo4j node count by label
docker exec psychology-neo4j cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN labels(n) as label, count(*) as count ORDER BY count DESC"

# Run detailed verification
docker exec psychology-backend python -m scripts.verify_graph
```

## 🚀 Quick Start (Full Setup)

```bash
# Clone và cài đặt nhanh

git clone https://github.com/linkoc119/psy-consulting-graphrag.git
cd psy-consulting-graphrag
# Backend reads configuration from backend/.env.


# Start infrastructure first
docker compose up -d ollama qdrant neo4j

# GPU optional
# docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama qdrant neo4j

# Hybrid optional: retrieval/backend CPU, Ollama LLM GPU
# docker compose -f docker-compose.yml -f docker-compose.hybrid.yml up -d ollama qdrant neo4j

docker exec psychology-ollama ollama pull qwen2.5:3b-instruct-q6_K

# Start backend and frontend
docker compose up -d --build

# GPU optional
# docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Hybrid optional
# docker compose -f docker-compose.yml -f docker-compose.hybrid.yml up -d --build

# Wait until backend is ready before indexing
curl http://localhost:8000/health/ready

# Run chunking and indexing
docker exec psychology-backend python -m scripts.index_data --clear-db --rechunk

# Check health
curl http://localhost:8000/health

# Open UI
open http://localhost:3000  # macOS
# or
start http://localhost:3000  # Windows
```

## 📁 Cấu trúc project

```
psy-consulting-graphrag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration
│   │   ├── models/              # Pydantic schemas
│   │   ├── routers/             # API endpoints
│   │   ├── services/            # Business logic
│   │   │   ├── llm_service.py       # LLM provider integration
│   │   │   ├── embedding_service.py # Vietnamese embedding
│   │   │   ├── reranker_service.py  # Vietnamese reranker
│   │   │   ├── qdrant_service.py    # Vector DB ops
│   │   │   ├── neo4j_service.py     # Graph DB ops
│   │   │   ├── prompt_builder.py    # Prompt construction
│   │   │   └── rag_service.py       # Core GraphRAG logic
│   │   └── utils/
│   │       └── prompts.py       # Prompt templates
│   ├── chunking/
│   │   └── chunk_processor.py   # PDF chunking
│   ├── data/                    # Processed chunks
│   ├── resources/               # PDF documents
│   │   ├── pdtt.pdf
│   │   ├── sctl.pdf
│   │   └── tvtl.pdf
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatInterface.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── index.jsx
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml              # CPU/default deployment
├── docker-compose.gpu.yml          # Optional NVIDIA GPU override
├── docker-compose.hybrid.yml       # Retrieval CPU + Ollama GPU override
└── README.md
```

## 🔍 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | API info |
| `/health` | GET | Deep health check |
| `/health/ready` | GET | Lightweight readiness check |
| `/chat/completion` | POST | Send chat message |
| `/chat/conversation/{id}` | GET | Get conversation history |
| `/chat/conversation/{id}` | DELETE | Delete conversation |
| `/chat/clear` | POST | Clear all conversations |


## 📝 Development

### Backend development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend development

```bash
cd frontend
npm install
npm start
```

### Debug logs

```bash
# Backend logs
docker compose logs -f backend

# Frontend logs
docker compose logs -f frontend
```

## ⚠️ Lưu ý quan trọng

1. **Không phải chẩn đoán y khoa**: Chatbot chỉ cung cấp hỗ trợ ban đầu, không thay thế bác sĩ/chuyên gia
2. **Dữ liệu nhạy cảm**: Không lưu trữ PII (Personally Identifiable Information)
3. **Red flags**: Luôn chuyển tuyến khi phát hiện nguy cơ tự sát, bạo lực, ảo giác
4. **Thuốc**: Tuyệt đối không kê đơn, chỉ dùng để nhận diện tiền sử bệnh
5. **Văn hóa**: Phản hồi phải phù hợp với văn hóa Việt Nam và lứa tuổi
6. **Performance:**
   - Backend startup: 5-10 phút (pre-load ML models)
   - Response time phụ thuộc vào LLM provider, CPU/GPU mode và retrieval/reranker (20-30s)
   - GPU auto-detection cho reranker

## 📄 License

[MIT License](LICENSE) - Dự án mang tính minh họa cho mục đích học thuật.

## 🙏 Credits

Thiết kế dựa trên:
- Bộ Y tế - Phác đồ Tâm thần 2020
- WHO - Sơ cứu Tâm lý (PFA)
- Bộ Giáo dục - Sổ tay Tư vấn Tâm lý Học đường
- AITeamVN - Vietnamese Embedding & Reranker models
