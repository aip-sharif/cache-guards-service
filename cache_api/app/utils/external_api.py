# external_api.py
import httpx
from fastapi import HTTPException
from ..config import EXTERNAL_API_URL, SC_GATEWAY_ADMIN_KEY


async def call_external_cache_api(name: str) -> str:
    """
    POST {EXTERNAL_API_URL}
    Content-Type: application/json
    Body: {"name": "..."}

    پاسخ (201): {"project_id": "...", "api_key": "..."}
    """
    payload = {"name": name}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                EXTERNAL_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {SC_GATEWAY_ADMIN_KEY}",
                    "Content-Type": "application/json"
                    },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"External cache API returned error: {e.response.status_code}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach external cache API: {e}",
        )

    api_key = data.get("api_key")
    project_id = data.get("project_id")
    if not api_key:
        raise HTTPException(
            status_code=502,
            detail="External cache API response did not contain 'api_key'",
        )
    if not project_id:
        raise HTTPException(
            status_code=502,
            detail="External cache API response did not contain 'project_id'",
        )

    return api_key, project_id