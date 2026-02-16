import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.models import Base
from app.inference import inference_client
from app.routers import upload, sessions, database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("Starting Posture Analysis API...")
    # Create tables (in production, use Alembic migrations)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified.")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await inference_client.close()
    await engine.dispose()
    logger.info("Cleanup completed.")


app = FastAPI(
    title="Student Posture Analysis API",
    description="AI-based student posture analysis from CCTV images/videos",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(database.router, prefix="/api", tags=["Database Explorer"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "posture-analysis-api"}
