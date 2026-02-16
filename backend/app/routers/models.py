import logging
from fastapi import APIRouter, HTTPException

from app.inference import inference_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models():
    """Proxy model metadata from the model server."""
    try:
        return await inference_client.get_model_info()
    except Exception as e:
        logger.exception("Failed to fetch model info from model server")
        raise HTTPException(status_code=502, detail=f"Model server error: {e}")
