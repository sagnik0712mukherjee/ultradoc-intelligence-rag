"""
Upload API Module
"""

from fastapi import APIRouter, UploadFile, File, Form
from typing import Dict, List
import shutil
import os

from src.core.document_processor import DocumentProcessor
from src.core.chunker import StructureAwareChunker
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import VectorStore

import src.core.app_state as app_state

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), api_key: str = Form(...)
) -> Dict[str, str]:
    """
    Upload and process document.

    Args:
        file (UploadFile): Uploaded document.
        api_key (str): OpenAI API key.

    Returns:
        Dict[str, str]: Status message.
    """

    temp_file_path: str = f"temp_{file.filename}"

    # Save uploaded file temporarily
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract document text
    processor: DocumentProcessor = DocumentProcessor()

    document_text: str = processor.extract_text(temp_file_path)

    # Store document text in shared state
    app_state.DOCUMENT_TEXT_STORAGE = document_text

    # Chunk document
    chunker: StructureAwareChunker = StructureAwareChunker()

    chunks: List[Dict[str, str]] = chunker.chunk_document(document_text)

    # Generate embeddings
    embedding_service: EmbeddingService = EmbeddingService(api_key)

    chunk_texts: List[str] = [chunk.get("content") for chunk in chunks]

    print("\n\nNumber of chunks:", len(chunks))
    print("\n\nFirst chunk:", chunks[0])

    embeddings: List[List[float]] = embedding_service.generate_embeddings_batch(
        chunk_texts
    )

    # Initialize vector store
    embedding_dimension: int = len(embeddings[0])

    app_state.VECTOR_STORE_INSTANCE = VectorStore(embedding_dimension)

    # Add vectors to shared vector store
    app_state.VECTOR_STORE_INSTANCE.add_vectors(embeddings, chunks)
    print("\n\nTotal vectors in index:", app_state.VECTOR_STORE_INSTANCE.index.ntotal)

    # Reset memory on new upload
    app_state.MEMORY_MANAGER_INSTANCE.clear_memory()

    # Remove temporary file
    os.remove(temp_file_path)

    return {"status": "Document uploaded and indexed successfully."}
