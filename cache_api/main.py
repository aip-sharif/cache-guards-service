# main.py

import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from contextlib import asynccontextmanager

from app.utils.db_utils import create_db_and_tables
from app.routes.routes_cache import router as cache_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(
    title="Cache Service API",
    version="1.0.0",
    lifespan=lifespan,
)

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


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    is_dev = os.getenv("SC_ENVIRONMENT", "production").lower() == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)