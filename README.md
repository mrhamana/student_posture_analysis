# Student Posture Analysis System

AI-powered web application for analyzing student postures from CCTV images and videos. Uses YOLO + BoT-SORT + posture classifier pipeline to detect, track, and classify student postures in real-time.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│   React UI   │◄──►│  FastAPI      │◄──►│  Inference API   │
│  (TypeScript) │    │  Backend     │    │  (YOLO+BoTSORT)  │
└──────────────┘    └──────┬───────┘    └──────────────────┘
                           │
                    ┌──────▼───────┐
                    │  PostgreSQL  │
                    └──────────────┘
```

## Features

- **Media Upload**: Image (JPEG, PNG, WebP) and video (MP4, AVI, MKV, MOV) upload with drag & drop
- **Real-time Processing**: Async video processing with live progress updates
- **Detection Visualization**: Canvas-based bounding box overlays with student IDs, posture labels, and confidence scores
- **Sortable Data Table**: Filter by posture class, search by student ID, sortable columns
- **Session Summary**: Posture distribution charts, statistics, per-student breakdown
- **Session History**: Browse and revisit previous analysis sessions
- **Error Handling**: Retry with exponential backoff, graceful error states, validation

## Posture Classes

| Class                | Description                      |
| -------------------- | -------------------------------- |
| Listening            | Student is actively listening    |
| Looking Screen       | Student is looking at screen     |
| Hand Raising         | Student has hand raised          |
| Sleeping / Head Down | Student is sleeping or head down |
| Turning Back         | Student is turned around         |
| Standing             | Student is standing              |
| Writing              | Student is writing               |
| Reading              | Student is reading               |

## API Endpoints

| Method | Endpoint                            | Description                     |
| ------ | ----------------------------------- | ------------------------------- |
| POST   | `/api/upload`                       | Upload image/video for analysis |
| GET    | `/api/sessions`                     | List all sessions               |
| GET    | `/api/session/{id}`                 | Get session details             |
| GET    | `/api/session/{id}/progress`        | Get processing progress         |
| GET    | `/api/session/{id}/students`        | Get detected students           |
| GET    | `/api/session/{id}/posture-summary` | Get posture analysis summary    |
| GET    | `/api/session/{id}/records`         | Get latest posture records      |
| DELETE | `/api/session/{id}`                 | Delete a session                |
| GET    | `/health`                           | Health check                    |

## Quick Start (Docker Compose)

```bash
# 1. Clone and navigate
cd semester_project

# 2. Copy environment file
cp .env.example .env

# 3. Edit .env — set your inference API URL
#    INFERENCE_API_URL=http://your-model-server:8001/predict

# 4. Start all services
docker compose up --build

# 5. Open browser
#    http://localhost
```

## Manual Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 16+

### Database

```bash
# Create database
createdb posture_analysis

# Apply schema
psql -d posture_analysis -f backend/schema.sql
```

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure env
cp .env.example .env
# Edit .env with your settings

# Run
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run dev server (proxies /api to backend)
npm run dev
```

Open http://localhost:5173

## Project Structure

```
semester_project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app, lifespan, CORS
│   │   ├── config.py         # Settings from env
│   │   ├── database.py       # Async SQLAlchemy engine
│   │   ├── models.py         # ORM models (Session, Student, PostureRecord)
│   │   ├── schemas.py        # Pydantic request/response schemas
│   │   ├── services.py       # Business logic & DB operations
│   │   ├── inference.py      # Inference API client with retry
│   │   └── routers/
│   │       ├── upload.py     # POST /upload endpoint
│   │       └── sessions.py   # GET session/student/summary endpoints
│   ├── schema.sql            # PostgreSQL DDL
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx           # Root component
│   │   ├── index.css         # Tailwind CSS
│   │   ├── types.ts          # TypeScript types
│   │   ├── api/
│   │   │   └── client.ts     # Axios API client
│   │   ├── hooks/
│   │   │   └── useAppState.ts # State management (useReducer)
│   │   └── components/
│   │       ├── FileUpload.tsx       # Drag & drop upload
│   │       ├── DetectionCanvas.tsx  # Canvas overlay rendering
│   │       ├── PostureTable.tsx     # Sortable/filterable table
│   │       ├── SessionSummary.tsx   # Stats + bar chart
│   │       ├── ProgressBar.tsx      # Processing progress
│   │       └── SessionList.tsx      # Session history sidebar
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
├── nginx.conf
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

## Inference API Contract

The system expects your inference API to accept multipart file upload and return JSON.

### Image Mode

**Request**: `POST /predict?mode=image` with multipart file

**Response**:

```json
{
  "frame_id": 0,
  "students": [
    {
      "id": 3,
      "bbox": [120, 80, 280, 350],
      "posture": "Listening",
      "confidence": 0.93
    }
  ]
}
```

### Video Mode

**Request**: `POST /predict?mode=video` with multipart file

**Response**: Newline-delimited JSON (streamed), one frame per line:

```
{"frame_id": 0, "students": [{"id": 1, "bbox": [100, 50, 250, 300], "posture": "Writing", "confidence": 0.87}]}
{"frame_id": 1, "students": [{"id": 1, "bbox": [102, 51, 252, 302], "posture": "Writing", "confidence": 0.89}]}
```

## Database Schema

- **sessions**: UUID PK, filename, media_type, status, frame counts, timestamps
- **students**: Auto-increment PK, session FK, tracker_id (BoT-SORT), posture change count
- **posture_records**: Auto-increment PK, student FK, frame_id, posture enum, confidence, bbox coords

## Key Design Decisions

1. **Async throughout**: FastAPI async endpoints, SQLAlchemy async sessions, httpx async client
2. **Background processing**: Upload returns immediately; processing runs in background tasks
3. **Batch DB writes**: Posture records are batch-inserted per frame, committed every 10 frames for video
4. **Connection pooling**: SQLAlchemy pool with 20 connections, 10 overflow
5. **Retry with backoff**: Inference API calls retry 3x with exponential backoff
6. **Canvas rendering**: Bounding box overlays use HTML5 Canvas for 60fps performance
7. **useReducer state**: Centralized state management without external dependencies
8. **Polling progress**: Frontend polls `/progress` every 2s during processing

## License

MIT
