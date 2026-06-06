from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.runtime import ChatRuntime
from backend.settings import BackendSettings


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)


def create_app() -> FastAPI:
    settings = BackendSettings.from_env()
    runtime = ChatRuntime(settings)
    app = FastAPI(title="VN Stock MAO ARAG Backend", version="0.1.0")
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return runtime.health_payload()

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return runtime.config_payload()

    @app.post("/api/chat")
    def api_chat(payload: ChatRequest) -> JSONResponse:
        try:
            result = runtime.chat(payload.message, history=payload.history)
            return JSONResponse(content=result)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:
            logging.exception("Chat request failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    if settings.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")

    @app.exception_handler(HTTPException)
    def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    return app


app = create_app()

