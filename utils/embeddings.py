from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore():
    embeddings = get_embeddings()
    index = Path(VECTORSTORE_PATH) / "index.faiss"
    if index.exists():
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


def save_vectorstore(chunks, metadatas=None):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(
        chunks,
        embeddings,
        metadatas=metadatas,
    )
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore