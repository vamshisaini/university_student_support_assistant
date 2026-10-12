
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

FAISS_DIR = BASE_DIR / "vectorstores" / "faiss_index"
CHROMA_DIR = BASE_DIR / "vectorstores" / "chroma_db"

EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------
# 2. STREAMLIT PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="JNTUH Academic Support Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 JNTUH Academic Support Assistant")

st.write(
    "Ask questions about JNTUH academic regulations, "
    "syllabus information, and academic requirements."
)


# --------------------------------------------------
# 3. LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_NAME
    )


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_NAME)


# --------------------------------------------------
# 4. LOAD PERSISTENT VECTOR DATABASES
# --------------------------------------------------

@st.cache_resource
def load_faiss():
    if not (FAISS_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "FAISS index not found. Run the notebook "
            "to create the index first."
        )

    return FAISS.load_local(
        str(FAISS_DIR),
        load_embedding_model(),
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_chroma():
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            "Chroma database not found. Run the notebook "
            "to create the database first."
        )

    return Chroma(
        collection_name="jntuh_documents",
        embedding_function=load_embedding_model(),
        persist_directory=str(CHROMA_DIR)
    )


# --------------------------------------------------
# 5. RAG PROMPT AND CHAIN
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a JNTUH Academic Support Assistant.

Answer the question using only the provided context.
Do not invent academic rules, dates, or requirements.
If the answer is not available in the context,
clearly say that you could not find it in the provided documents.

Context:
{context}

Question:
{question}

Answer:
""")

output_parser = StrOutputParser()


# --------------------------------------------------
# 6. BASIC RAG - FAISS
# --------------------------------------------------

def faiss_rag(question):
    vector_store = load_faiss()

    docs = vector_store.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    answer = (
        prompt
        | load_llm()
        | output_parser
    ).invoke({
        "context": context,
        "question": question
    })

    return answer, docs


# --------------------------------------------------
# 7. BASIC RAG - CHROMA
# --------------------------------------------------

def chroma_rag(question):
    vector_store = load_chroma()

    docs = vector_store.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    answer = (
        prompt
        | load_llm()
        | output_parser
    ).invoke({
        "context": context,
        "question": question
    })

    return answer, docs


# --------------------------------------------------
# 8. ADVANCED RAG - FAISS + RERANKING
# --------------------------------------------------

def advanced_rag(question):
    vector_store = load_faiss()

    # Retrieve candidate chunks
    candidate_docs = vector_store.similarity_search(
        question,
        k=15
    )

    if not candidate_docs:
        return (
            "No relevant document chunks were found.",
            []
        )

    # Score each question-document pair
    pairs = [
        [question, doc.page_content]
        for doc in candidate_docs
    ]

    scores = load_reranker().predict(pairs)

    # Rank chunks by relevance
    ranked_docs = sorted(
        zip(candidate_docs, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # Keep the four highest-ranked chunks
    docs = [
        doc for doc, score in ranked_docs[:4]
    ]

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    answer = (
        prompt
        | load_llm()
        | output_parser
    ).invoke({
        "context": context,
        "question": question
    })

    return answer, docs


# --------------------------------------------------
# 9. SELECT RAG VERSION
# --------------------------------------------------

st.sidebar.header("RAG Configuration")

rag_version = st.sidebar.selectbox(
    "Choose a RAG approach",
    [
        "Basic RAG - FAISS",
        "Basic RAG - ChromaDB",
        "Advanced RAG - Reranking"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("**Embedding model**")
st.sidebar.caption("Sentence Transformers: all-MiniLM-L6-v2")

st.sidebar.write("**Answer generation**")
st.sidebar.caption("Google Gemini")


# --------------------------------------------------
# 10. QUESTION INPUT
# --------------------------------------------------

question = st.text_area(
    "Enter your academic question",
    placeholder=(
        "Example: What are the attendance "
        "requirements for B.Tech students?"
    ),
    height=100
)

ask_button = st.button(
    "Get Answer",
    type="primary"
)


# --------------------------------------------------
# 11. RUN THE SELECTED PIPELINE
# --------------------------------------------------

if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")

    elif not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "Google API key not found. "
            "Add GOOGLE_API_KEY to your .env file."
        )

    else:
        try:
            start_time = time.time()

            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                if rag_version == "Basic RAG - FAISS":
                    answer, sources = faiss_rag(question)

                elif rag_version == "Basic RAG - ChromaDB":
                    answer, sources = chroma_rag(question)

                else:
                    answer, sources = advanced_rag(question)

            elapsed_time = time.time() - start_time

            st.subheader("Answer")
            st.markdown(answer)

            st.caption(
                f"RAG version: {rag_version} | "
                f"Response time: {elapsed_time:.2f} seconds"
            )

            st.subheader("Retrieved Sources")

            if sources:
                for i, doc in enumerate(sources, start=1):
                    with st.expander(f"Source {i}"):

                        st.write(
                            "**File:**",
                            Path(
                                doc.metadata.get(
                                    "source", "Unknown"
                                )
                            ).name
                        )

                        page = doc.metadata.get("page")

                        if page is not None:
                            st.write(
                                "**Page:**",
                                int(page) + 1
                            )

                        st.write(doc.page_content)

            else:
                st.info("No source documents were retrieved.")

        except Exception as error:
            st.error(
                "Something went wrong while processing "
                "your question."
            )
            st.exception(error)