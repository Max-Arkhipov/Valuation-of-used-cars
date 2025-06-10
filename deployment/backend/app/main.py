import time
import uuid

from app.utils import setup_file_logger
logger = setup_file_logger()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from app.routers import api

app = FastAPI(
    title="ML Model Management API",
    docs_url="/api/openapi",
    version="1.0"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception:
        raise

    elapsed = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.2f}ms)",
        extra={"request_id": request_id}
    )
    response.headers["X-Request-ID"] = request_id
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"[{request_id}] Необработанное исключение: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера", "request_id": request_id},
    )

app.include_router(api.router, prefix="/api/v1", tags=["ML API"])

@app.on_event("startup")
async def startup_event():
    logger.info("Приложение FastAPI запущено")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Приложение FastAPI завершает работу")

@app.get("/", tags=["Проверка работоспособности"])
def read_root():
    logger.info("Запрошена проверка работоспособности")
    return {"message": "ML API запущен!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
