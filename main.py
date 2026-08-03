# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.utils.db_utils import create_db_and_tables
from app.routes.routes_cache import router as cache_router
import uvicorn

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


@app.get("/")
def root():
    return {"status": "ok", "message": "Cache Service is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
