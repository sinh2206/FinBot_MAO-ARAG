from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(title="MAO-ARAG API")

# Serve frontend assets.
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
# Serve generated charts and persisted files.
app.mount("/storage", StaticFiles(directory="storage"), name="storage")


@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
