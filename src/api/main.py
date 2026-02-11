"""
Main FastAPI Application
"""

from fastapi import FastAPI
from src.api.upload import router as upload_router
from src.api.ask import router as ask_router
from src.api.extract import router as extract_router

app = FastAPI(title="UltraDoc Intelligence RAG API")

app.include_router(upload_router)

app.include_router(ask_router)

app.include_router(extract_router)
