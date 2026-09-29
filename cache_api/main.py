# main.py

import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from contextlib import asynccontextmanager
from sqlalchemy import text

from app.utils.db_utils import create_db_and_tables, engine
from app.routes.routes_cache import router as cache_router
from app.rate_limit import RateLimitMiddleware, BodySizeLimitMiddleware
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger("main")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [A03 FIX] دیگه به‌صورت پیش‌فرض create_all اجرا نمی‌شه - schema فقط
    # از طریق Alembic migration تغییر می‌کنه (کنترل‌شده، با تاریخچه و
    # امکان rollback)، نه هر بار که سرویس بالا میاد بی‌سروصدا.
    # فقط توی dev محلی (SC_ENVIRONMENT=development) خودکار اجرا می‌شه.
    is_dev = os.getenv("SC_ENVIRONMENT", "production").lower() == "development"
    if is_dev:
        logger.warning(
            "SC_ENVIRONMENT=development: running create_all() at startup. "
            "In production, run `alembic upgrade head` as a deploy step instead."
        )
        create_db_and_tables()
    yield


app = FastAPI(
    title="Cache Service API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow requests from any frontend host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [A09 FIX] محدودیت سایز body + rate limit پایه
app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=1 * 1024 * 1024)  # 1 MiB
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

app.include_router(cache_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "message": str(exc.detail),
            "data": None,
        },
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Cache Service is running"}


# [F4 FIX] /health قبلاً همیشه 200 می‌داد، حتی وقتی Postgres پایین بود - پس
# readiness probe ترافیک رو به پادی می‌فرستاد که نمی‌تونه جواب بده.
# الان /health (readiness) دیتابیس رو چک می‌کنه و در صورت خطا 503 می‌ده؛
# /live (liveness) فقط می‌گه پروسه بالاست - liveness نباید به دیتابیس
# وابسته باشه، وگرنه قطعی Postgres باعث ری‌استارت پشت‌سرهم پادها می‌شه.
@app.get("/live")
def liveness_check():
    return {"status": "alive"}


@app.get("/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        logger.exception("health check: database unreachable")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "unreachable"},
        )
    return {"status": "healthy", "database": "ok"}


if __name__ == "__main__":
    import uvicorn

    is_dev = os.getenv("SC_ENVIRONMENT", "production").lower() == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)