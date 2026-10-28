# rate_limit.py
"""
[A09 FIX - نسخه‌ی پایه، بدون وابستگی خارجی]

دو میان‌افزار ساده:
  1) BodySizeLimitMiddleware: رد کردن body های خیلی بزرگ قبل از این‌که
     اصلاً وارد پردازش JSON/pydantic بشن.
  2) RateLimitMiddleware: محدودیت تعداد درخواست در بازه‌ی زمانی، به‌ازای
     هر کلاینت (بر اساس IP یا هدر Authorization اگه موجود باشه).

⚠️ محدودیت شناخته‌شده: این پیاده‌سازی in-memory هست - یعنی اگه سرویس
چند replica اجرا بشه (چند کانتینر/پاد پشت لودبالانسر)، هر replica
شمارنده‌ی جدای خودش رو داره (rate limit واقعی مؤثر می‌شه N برابر تعداد
replica ها). برای production واقعی با چند replica، باید این شمارنده رو
به Redis منتقل کرد (مثلاً با الگوریتم sliding-window-log روی Redis).
این نسخه برای یه instance تنها یا به‌عنوان یه لایه‌ی دفاعی اول کافیه.
"""

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_bytes: int = 1 * 1024 * 1024):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "status_code": 413,
                            "message": f"Request body too large (max {self.max_body_bytes} bytes)",
                            "data": None,
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # اول از هدر Authorization (اگه بود، محدودیت به‌ازای هر کلید/توکنه،
        # نه IP - چند کاربر پشت یه NAT رو تحت تأثیر قرار نمی‌ده)
        auth = request.headers.get("authorization")
        if auth:
            return f"auth:{auth[-16:]}"  # فقط ۱۶ کاراکتر آخر - نه کل توکن، برای جلوگیری از نگه‌داری کامل secret در حافظه
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        key = self._client_key(request)
        now = time.monotonic()
        window_start = now - self.window_seconds

        hits = self._hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "status_code": 429,
                    "message": "Too many requests, please slow down.",
                    "data": None,
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        hits.append(now)
        return await call_next(request)
