# guard_warmup.py
"""
وقتی یه کش با guard فعال ثبت/ویرایش می‌شه، gateway ایندکس guard (امبدینگ
همه‌ی exemplar های policy) رو تنبل و فقط موقع «اولین درخواست واقعی» می‌سازه -
یعنی اولین کاربر واقعی هزینه‌ی اون ساخت رو می‌ده.

اینجا بعد از نوشتن رکورد توی دیتابیس، یه درخواست chat/completions به gateway
می‌زنیم تا اون ساخت همون‌جا انجام بشه:

    POST {SC_GATEWAY_CHAT_URL}
    Authorization: Bearer <cache_key>      # همون api_key ای که موقع register ساخته شد
    {"model": "<llm_model>", "messages": [{"role": "user", "content": "..."}]}

نکته: این درخواست fire-and-forget هست - هر خطایی فقط لاگ می‌شه و هیچ‌وقت
باعث شکست register/edit نمی‌شه، چون رکورد از قبل با موفقیت commit شده.
"""

import logging

import httpx

from ..config import (
    GUARD_WARMUP_ENABLED,
    GUARD_WARMUP_PROMPT,
    GUARD_WARMUP_TIMEOUT,
    SC_GATEWAY_CHAT_URL,
)

logger = logging.getLogger("guard")


async def warmup_guard(cache_key: str, model: str) -> None:
    """
    gateway رو با کلید خودِ پروژه صدا می‌زنه تا ایندکس guard ساخته بشه.
    هیچ‌وقت exception نمی‌ندازه.
    """
    if not GUARD_WARMUP_ENABLED:
        logger.info("guard warmup skipped (disabled by config)")
        return

    if not SC_GATEWAY_CHAT_URL:
        logger.warning("guard warmup skipped: SC_GATEWAY_CHAT_URL is not configured")
        return

    if not cache_key or not model:
        logger.warning("guard warmup skipped: missing cache_key or model")
        return

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": GUARD_WARMUP_PROMPT}],
    }

    try:
        async with httpx.AsyncClient(timeout=GUARD_WARMUP_TIMEOUT) as client:
            response = await client.post(
                SC_GATEWAY_CHAT_URL,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cache_key}",
                },
            )
    except httpx.TimeoutException:
        logger.warning("guard warmup timed out after %ss (model=%s)", GUARD_WARMUP_TIMEOUT, model)
        return
    except httpx.RequestError as e:
        logger.warning("guard warmup could not reach the gateway: %s", e)
        return

    # ۴۰۳ یعنی خودِ guard جلوی پیام رو گرفته - یعنی ایندکس ساخته شد و کار کرد،
    # پس این هم یه warmup موفق حساب می‌شه.
    if response.status_code >= 400:
        logger.warning(
            "guard warmup finished with HTTP %s (model=%s): %s",
            response.status_code,
            model,
            response.text[:500],
        )
    else:
        logger.info("guard warmup succeeded (model=%s)", model)
