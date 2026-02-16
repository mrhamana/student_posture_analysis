from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Database (SQLite for local development)
    DATABASE_URL: str = "sqlite+aiosqlite:///./posture_analysis.db"

    # Inference API (model_server)
    INFERENCE_API_URL: str = "http://localhost:8001/predict"
    INFERENCE_API_TIMEOUT: int = 60

    # Upload
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/webp"
    ALLOWED_VIDEO_TYPES: str = "video/mp4,video/avi,video/x-matroska,video/quicktime"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def allowed_image_types_list(self) -> List[str]:
        return [t.strip() for t in self.ALLOWED_IMAGE_TYPES.split(",")]

    @property
    def allowed_video_types_list(self) -> List[str]:
        return [t.strip() for t in self.ALLOWED_VIDEO_TYPES.split(",")]

    @property
    def allowed_types_list(self) -> List[str]:
        return self.allowed_image_types_list + self.allowed_video_types_list

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
