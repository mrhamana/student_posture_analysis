import httpx
import logging
from typing import List, AsyncIterator
from app.config import settings
from app.schemas import InferenceFrameResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0


class InferenceClient:
    """Client for communicating with the pretrained inference API."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=settings.INFERENCE_API_TIMEOUT,
                    write=30.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _retry_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute request with exponential backoff retry."""
        import asyncio

        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                client = await self._get_client()
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as e:
                last_exception = e
                wait_time = BACKOFF_BASE**attempt
                logger.warning(
                    f"Inference API request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)

        logger.error(f"Inference API request failed after {MAX_RETRIES} attempts")
        raise last_exception  # type: ignore

    async def analyze_image(
        self, file_content: bytes, filename: str
    ) -> InferenceFrameResult:
        """Send a single image to the inference API."""
        files = {"file": (filename, file_content)}
        response = await self._retry_request(
            "POST",
            settings.INFERENCE_API_URL,
            files=files,
            params={"mode": "image"},
        )
        data = response.json()
        return InferenceFrameResult(**data)

    async def analyze_video_stream(
        self, file_content: bytes, filename: str
    ) -> AsyncIterator[InferenceFrameResult]:
        """Send video to inference API and stream frame-by-frame results."""
        import json

        files = {"file": (filename, file_content)}
        client = await self._get_client()

        try:
            async with client.stream(
                "POST",
                settings.INFERENCE_API_URL,
                files=files,
                params={"mode": "video"},
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,  # Videos can take long
                    write=120.0,
                    pool=10.0,
                ),
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    # Parse newline-delimited JSON
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                yield InferenceFrameResult(**data)
                            except (json.JSONDecodeError, Exception) as e:
                                logger.warning(f"Failed to parse frame result: {e}")
                                continue

                # Handle remaining buffer
                if buffer.strip():
                    try:
                        data = json.loads(buffer.strip())
                        yield InferenceFrameResult(**data)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to parse final frame result: {e}")

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error(f"Video streaming inference failed: {e}")
            raise


inference_client = InferenceClient()
