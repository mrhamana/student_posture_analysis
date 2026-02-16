# Student Posture Analysis System

An AI-powered web application that detects, tracks, and classifies student postures from classroom CCTV images and videos. Built with a **YOLO + BoT-SORT** object-tracking pipeline on the backend and a **React + TypeScript** frontend for real-time visualization and analysis.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Posture Classes](#posture-classes)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Setup](#quick-setup)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
  - [Backend REST API](#backend-rest-api)
  - [Model Server Inference API](#model-server-inference-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [Technical Details](#technical-details)

---

## Overview

This system allows educators and administrators to upload classroom footage (images or videos) and receive detailed posture analysis for each detected student. The pipeline works as follows:

1. **Upload** — The user uploads an image or video through the web interface.
2. **Detect & Track** — The model server runs YOLOv8 with BoT-SORT multi-object tracking to identify and follow individual students across frames.
3. **Classify** — Each detected student is classified into one of eight posture categories.
4. **Visualize** — Results are rendered as interactive bounding-box overlays, sortable data tables, and summary charts.

---

## Architecture

```
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────────┐
│   React + Vite  │  HTTP  │   FastAPI        │  HTTP  │   Model Server      │
│   Frontend      │◄──────►│   Backend        │◄──────►│   (YOLO + BoT-SORT) │
│   :5173         │        │   :8000          │        │   :8001              │
└─────────────────┘        └────────┬─────────┘        └─────────────────────┘
                                    │
                            ┌───────▼────────┐
                            │   SQLite /     │
                            │   PostgreSQL   │
                            └────────────────┘
```

| Service        | Technology                          | Port  |
| -------------- | ----------------------------------- | ----- |
| Frontend       | React 18, TypeScript, Tailwind CSS  | 5173  |
| Backend API    | FastAPI, SQLAlchemy (async), Pydantic | 8000  |
| Model Server   | FastAPI, Ultralytics YOLOv8, OpenCV | 8001  |
| Database       | SQLite (default) / PostgreSQL       | —     |

---

## Features

- **Drag & Drop Upload** — Supports JPEG, PNG, WebP images and MP4, AVI, MKV, MOV videos (up to 500 MB).
- **Real-time Progress** — Asynchronous video processing with live progress bar updates via polling.
- **Detection Visualization** — HTML5 Canvas overlay rendering bounding boxes with student IDs, posture labels, and confidence scores.
- **Sortable & Filterable Table** — Search by student ID, filter by posture class, sort by any column.
- **Session Summary** — Bar charts (Recharts) showing posture distribution, per-student breakdowns, and aggregate statistics.
- **Session History** — Sidebar listing all previous analysis sessions for easy revisit and comparison.
- **Demo Mode** — Model server generates deterministic fake detections when no YOLO weights are present, enabling frontend/backend development without a GPU.
- **Robust Error Handling** — Retry with exponential backoff on inference calls, graceful error states, and input validation throughout.

---

## Posture Classes

The model classifies each detected student into one of the following eight categories:

| Class                | Description                                   |
| -------------------- | --------------------------------------------- |
| Listening            | Actively paying attention to the instructor   |
| Looking Screen       | Eyes directed at a screen or monitor           |
| Hand Raising         | One or both hands raised                       |
| Sleeping / Head Down | Head resting on desk or lowered                |
| Turning Back         | Body or head turned away from the front        |
| Standing             | Student is standing up                         |
| Writing              | Actively writing or taking notes               |
| Reading              | Looking down at a book or reading material     |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 20+** and npm
- **Git**

> **GPU (optional):** A CUDA-capable GPU accelerates inference. Without one, the model server falls back to CPU or demo mode.

### Quick Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd semester_project

# 2. Install all dependencies (backend + model server + frontend)
bash setup.sh

# 3. Start all three services
bash run.sh
```

This launches:

| Service      | URL                        |
| ------------ | -------------------------- |
| Frontend     | http://localhost:5173      |
| Backend API  | http://localhost:8000      |
| Model Server | http://localhost:8001      |

Press `Ctrl+C` to stop all services.

### Manual Setup

<details>
<summary><strong>Model Server</strong></summary>

```bash
cd model_server

# Install dependencies
pip install -r requirements.txt

# Place your YOLO weights at models/best.pt (optional — demo mode works without it)

# Start the server
python server.py
# → http://localhost:8001
```

**Environment variables:**

| Variable      | Default                    | Description                      |
| ------------- | -------------------------- | -------------------------------- |
| `MODEL_PATH`  | `models/best.pt`           | Path to YOLO weights             |
| `SKIP_FRAMES` | `5`                        | Process every Nth frame in video |
| `PORT`        | `8001`                     | Server port                      |

</details>

<details>
<summary><strong>Backend</strong></summary>

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# (Optional) Create a .env file — see Configuration section below

# Start the server
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000
```

The database tables are created automatically on startup. By default, an SQLite file (`posture_analysis.db`) is used — no external database required.

</details>

<details>
<summary><strong>Frontend</strong></summary>

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server (proxies /api → localhost:8000)
npm run dev
# → http://localhost:5173
```

</details>

---

## Usage

1. Open **http://localhost:5173** in your browser.
2. **Upload** an image or video using the drag-and-drop zone or file picker.
3. For images, results appear immediately. For videos, a **progress bar** tracks frame-by-frame processing.
4. View **bounding boxes** overlaid on the media, showing each student's ID, posture, and confidence.
5. Use the **data table** to sort, filter, and search through detections.
6. Check the **summary panel** for posture distribution charts and per-student statistics.
7. Browse **session history** in the sidebar to revisit past analyses.

---

## API Reference

### Backend REST API

Base URL: `http://localhost:8000`

| Method   | Endpoint                            | Description                          |
| -------- | ----------------------------------- | ------------------------------------ |
| `POST`   | `/api/upload`                       | Upload an image or video for analysis |
| `GET`    | `/api/sessions`                     | List all analysis sessions           |
| `GET`    | `/api/session/{id}`                 | Get session metadata                 |
| `GET`    | `/api/session/{id}/progress`        | Get video processing progress        |
| `GET`    | `/api/session/{id}/students`        | Get detected students in a session   |
| `GET`    | `/api/session/{id}/posture-summary` | Get posture distribution summary     |
| `GET`    | `/api/session/{id}/records`         | Get latest posture records           |
| `DELETE` | `/api/session/{id}`                 | Delete a session and its data        |
| `GET`    | `/health`                           | Health check                         |

Interactive API docs are available at **http://localhost:8000/docs** (Swagger UI).

### Model Server Inference API

Base URL: `http://localhost:8001`

#### `POST /predict?mode=image`

Accepts a multipart file upload. Returns a single JSON object:

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

#### `POST /predict?mode=video`

Accepts a multipart file upload. Returns **newline-delimited JSON** (NDJSON), streamed one frame per line:

```
{"frame_id": 0, "students": [{"id": 1, "bbox": [100, 50, 250, 300], "posture": "Writing", "confidence": 0.87}]}
{"frame_id": 5, "students": [{"id": 1, "bbox": [102, 51, 252, 302], "posture": "Writing", "confidence": 0.89}]}
```

#### `GET /health`

Returns model status and current mode (`inference` or `demo`).

---

## Project Structure

```
semester_project/
├── backend/                         # FastAPI backend service
│   ├── app/
│   │   ├── main.py                  # App entry point, CORS, lifespan events
│   │   ├── config.py                # Pydantic settings (env vars)
│   │   ├── database.py              # Async SQLAlchemy engine & session
│   │   ├── models.py                # ORM models (Session, Student, PostureRecord)
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   ├── services.py              # Business logic & database operations
│   │   ├── inference.py             # HTTP client for model server (with retry)
│   │   └── routers/
│   │       ├── upload.py            # File upload & processing endpoint
│   │       └── sessions.py          # Session, student, and summary endpoints
│   ├── schema.sql                   # Reference PostgreSQL DDL
│   └── requirements.txt
│
├── frontend/                        # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx                  # Root component
│   │   ├── main.tsx                 # React entry point
│   │   ├── index.css                # Tailwind CSS styles
│   │   ├── types.ts                 # Shared TypeScript interfaces
│   │   ├── api/
│   │   │   └── client.ts            # Axios HTTP client
│   │   ├── hooks/
│   │   │   └── useAppState.ts       # Centralized state (useReducer)
│   │   └── components/
│   │       ├── FileUpload.tsx       # Drag & drop file upload
│   │       ├── DetectionCanvas.tsx  # Canvas bounding box overlay
│   │       ├── PostureTable.tsx     # Sortable/filterable data table
│   │       ├── SessionSummary.tsx   # Charts & statistics panel
│   │       ├── ProgressBar.tsx      # Video processing progress
│   │       └── SessionList.tsx      # Session history sidebar
│   ├── vite.config.ts               # Vite config with API proxy
│   ├── tailwind.config.js
│   └── package.json
│
├── model_server/                    # YOLO inference service
│   ├── server.py                    # FastAPI server with predict endpoint
│   ├── models/
│   │   └── best.pt                  # YOLO weights (not in repo)
│   └── requirements.txt
│
├── setup.sh                         # One-time dependency installation
├── run.sh                           # Start all services locally
└── README.md
```

---

## Configuration

The backend reads settings from environment variables (or a `.env` file in `backend/`):

| Variable                | Default                                                        | Description                           |
| ----------------------- | -------------------------------------------------------------- | ------------------------------------- |
| `DATABASE_URL`          | `sqlite+aiosqlite:///./posture_analysis.db`                    | Async database connection string      |
| `INFERENCE_API_URL`     | `http://localhost:8001/predict`                                | Model server prediction endpoint      |
| `INFERENCE_API_TIMEOUT` | `60`                                                           | Inference request timeout (seconds)   |
| `MAX_UPLOAD_SIZE_MB`    | `500`                                                          | Maximum upload file size in MB        |
| `ALLOWED_IMAGE_TYPES`   | `image/jpeg,image/png,image/webp`                              | Accepted image MIME types             |
| `ALLOWED_VIDEO_TYPES`   | `video/mp4,video/avi,video/x-matroska,video/quicktime`         | Accepted video MIME types             |
| `CORS_ORIGINS`          | `http://localhost:3000,http://localhost:5173`                   | Comma-separated allowed CORS origins  |

---

## Database Schema

The application uses three tables:

**`sessions`** — One row per uploaded file.

| Column            | Type        | Description                                    |
| ----------------- | ----------- | ---------------------------------------------- |
| `id`              | UUID (PK)   | Unique session identifier                      |
| `filename`        | Text        | Original uploaded filename                     |
| `media_type`      | Enum        | `image` or `video`                             |
| `status`          | Enum        | `pending` → `processing` → `completed`/`failed` |
| `total_frames`    | Integer     | Total frames in the media                      |
| `processed_frames`| Integer     | Frames processed so far                        |
| `created_at`      | Timestamp   | Upload time                                    |

**`students`** — One row per tracked individual per session.

| Column                 | Type        | Description                            |
| ---------------------- | ----------- | -------------------------------------- |
| `id`                   | Serial (PK) | Auto-increment ID                     |
| `session_id`           | UUID (FK)   | References `sessions.id`              |
| `tracker_id`           | Integer     | BoT-SORT assigned tracker ID          |
| `first_seen_frame`     | Integer     | First frame the student appears in    |
| `last_seen_frame`      | Integer     | Last frame the student appears in     |
| `total_posture_changes`| Integer     | Number of posture transitions         |

**`posture_records`** — One row per detection per frame.

| Column       | Type        | Description                                 |
| ------------ | ----------- | ------------------------------------------- |
| `id`         | Serial (PK) | Auto-increment ID                          |
| `student_id` | Integer (FK)| References `students.id`                   |
| `frame_id`   | Integer     | Frame number in the video                  |
| `posture`    | Enum        | One of the eight posture classes            |
| `confidence` | Float       | Model confidence score (0–1)               |
| `bbox_x1/y1/x2/y2` | Float | Bounding box coordinates             |

---

## Technical Details

| Aspect                  | Implementation                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| **Async I/O**           | FastAPI async endpoints, SQLAlchemy async sessions, httpx async HTTP client                      |
| **Background Tasks**    | Video processing runs in FastAPI background tasks; upload returns immediately                    |
| **Batch Writes**        | Posture records are batch-inserted per frame, committed every 10 frames                         |
| **Connection Pooling**  | SQLAlchemy pool with 20 connections, 10 overflow (PostgreSQL); WAL mode (SQLite)                 |
| **Retry Logic**         | Inference API calls retry 3× with exponential backoff                                           |
| **Canvas Rendering**    | Bounding boxes are drawn on HTML5 Canvas for high-performance overlay at 60fps                  |
| **State Management**    | `useReducer` for centralized state — no external state libraries                                |
| **Progress Polling**    | Frontend polls `/api/session/{id}/progress` every 2 seconds during video processing             |
| **Frame Skipping**      | Model server processes every Nth frame (default: 5) to balance speed and accuracy               |
