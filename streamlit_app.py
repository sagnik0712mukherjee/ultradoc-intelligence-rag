"""
Streamlit UI for UltraDoc Intelligence RAG System
"""

import streamlit as st
import requests
from typing import Dict
import os

# Backend URL - configurable for cloud deployment
API_BASE_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def upload_document(file, api_key: str) -> Dict:
    """
    Upload document to backend API.

    Args:
        file: Uploaded file object.
        api_key (str): OpenAI API key.

    Returns:
        Dict: API response.
    """

    files = {"file": file}

    data = {"api_key": api_key}

    response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)

    return response.json()


def ask_question(query: str, api_key: str) -> Dict:
    """
    Send question to backend API.

    Args:
        query (str): User query.
        api_key (str): OpenAI API key.

    Returns:
        Dict: API response.
    """

    response = requests.post(
        f"{API_BASE_URL}/ask", params={"query": query, "api_key": api_key}
    )

    return response.json()


def extract_structured_data(api_key: str) -> Dict:
    """
    Call structured extraction endpoint.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        Dict: Structured JSON.
    """

    response = requests.post(f"{API_BASE_URL}/extract", params={"api_key": api_key})

    return response.json()


def main() -> None:
    """
    Main Streamlit UI function.
    """

    st.set_page_config(page_title="UltraDoc Intelligence RAG", layout="wide")

    st.title("UltraDoc Intelligence – RAG System")

    st.sidebar.header("Configuration")

    api_key: str = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    uploaded_file = st.file_uploader(
        "Upload Logistics Document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"]
    )

    if uploaded_file and api_key:
        if st.button("Upload & Index Document"):
            with st.spinner("Processing document..."):
                response = upload_document(uploaded_file, api_key)

            st.success(response.get("status", "Upload completed."))

    st.divider()

    st.subheader("Ask Questions")

    query: str = st.text_input("Enter your question about the document")

    if st.button("Ask") and query and api_key:
        with st.spinner("Generating answer..."):
            response = ask_question(query, api_key)

        if "error" in response:
            st.error(response["error"])
        else:
            st.markdown("### Answer")
            st.write(response.get("answer"))

            st.markdown("### Confidence Score")
            st.write(round(response.get("confidence"), 4))

            st.markdown("### Supporting Sources")
            st.text_area("Sources", value=response.get("sources", ""), height=200)

    if st.sidebar.button("Clear Conversation Memory"):
        requests.post(f"{API_BASE_URL}/clear_memory")
        st.success("Memory cleared.")


if __name__ == "__main__":
    main()
