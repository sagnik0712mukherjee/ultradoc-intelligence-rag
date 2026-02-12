"""
Main FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.upload import router as upload_router
from src.api.ask import router as ask_router
from src.api.extract import router as extract_router

app = FastAPI(title="UltraDoc Intelligence RAG API")

# Add CORS middleware for cloud deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Streamlit Cloud
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)

app.include_router(ask_router)

app.include_router(extract_router)
