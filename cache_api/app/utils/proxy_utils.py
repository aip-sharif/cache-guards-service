# proxy_utils.py
"""
پروکسی مستقیم به mlops endpoint خودِ کاربر - با model و key ای که موقع
register ذخیره شده (llm_model/llm_key یا embedd_model/embedd_key).

اگه upstream خطا بده، دقیقاً با این envelope برمی‌گردونیم:
    {"error": {"message": "The upstream model provider returned an error (HTTP <code>).", "type": "gateway_error"}}
"""

import httpx
from fastapi import HTTPException


def _gateway_error(status_code: int, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": "gateway_error"}},
    )


async def forward_to_upstream(
    base_url: str,
    path: str,
    model: str,
    api_key: str,
    payload: dict,
    timeout: float = 60.0,
) -> dict:
    """
    payload رو با model/api_key خودِ پروژه به {base_url}{path} می‌فرسته
    و جواب JSON رو برمی‌گردونه. خطاهای upstream رو با envelope یکدست wrap می‌کنه.
    """
    if not base_url:
        raise _gateway_error(502, "Upstream base URL is not configured.")

    body = dict(payload)
    body["model"] = model  # مدل خودِ کاربر رو override می‌کنیم، حتی اگه چیز دیگه‌ای فرستاده باشه

    url = f"{base_url.rstrip('/')}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.TimeoutException:
        raise _gateway_error(504, "The upstream model provider timed out.")
    except httpx.RequestError as e:
        raise _gateway_error(502, f"The upstream model provider is unreachable: {e}")

    if response.status_code >= 400:
        raise _gateway_error(
            response.status_code,
            f"The upstream model provider returned an error (HTTP {response.status_code}).",
        )

    try:
        return response.json()
    except ValueError:
        raise _gateway_error(502, "The upstream model provider returned invalid JSON.")
