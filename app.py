import os
import re
import hashlib
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Information Retrieval and Hybrid RAG",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")

# Chroma database
DB_DIR = Path(r"C:\IR_PROJECT_CHROMA_DB")

# File index
INDEX_FILE = Path("indexed_files.json")

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SENSITIVE EXCEL COLUMNS
# =========================================================

SENSITIVE_WORDS = [
    "password",
    "passwd",
    "pass_word",
    "pwd",
    "secret",
    "login_password",
    "user_password"
]


def is_sensitive_column(column_name):
    """
    Check whether an Excel column contains sensitive information.
    """

    column_name = str(column_name).strip().lower()

    return any(
        word in column_name
        for word in SENSITIVE_WORDS
    )


# =========================================================
# FILE HASH
# =========================================================

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file.
    Used to avoid indexing the same unchanged file repeatedly.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# LOAD INDEXED FILES
# =========================================================

def load_indexed_files():

    if not INDEX_FILE.exists():
        return {}

    try:

        with open(
            INDEX_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# =========================================================
# SAVE INDEXED FILES
# =========================================================

def save_indexed_files(indexed_files):

    with open(
        INDEX_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            indexed_files,
            f,
            indent=4
        )


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return embeddings


# =========================================================
# LOAD CHROMA VECTOR DATABASE
# =========================================================

@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()

    vector_store = Chroma(
        collection_name="college_knowledge",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    return vector_store


# =========================================================
# EXCEL PROCESSING
# =========================================================

def process_excel(file_path):

    documents = []

    ignored_columns = []

    try:

        excel_data = pd.read_excel(
            file_path,
            sheet_name=None
        )

    except Exception as e:

        raise Exception(
            f"Could not read Excel file: {e}"
        )

    for sheet_name, df in excel_data.items():

        # Remove completely empty rows
        df = df.dropna(how="all")

        if df.empty:
            continue

        # Find sensitive columns
        columns_to_remove = []

        for column in df.columns:

            if is_sensitive_column(column):

                columns_to_remove.append(column)

                ignored_columns.append(
                    f"{file_path.name} → "
                    f"{sheet_name} → "
                    f"{column}"
                )

        # Remove sensitive columns
        if columns_to_remove:

            df = df.drop(
                columns=columns_to_remove
            )

        # Convert every row into a document
        for row_number, row in df.iterrows():

            row_data = []

            for column, value in row.items():

                if pd.notna(value):

                    row_data.append(
                        f"{column}: {value}"
                    )

            if not row_data:
                continue

            content = "\n".join(row_data)

            document = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_type": "Excel",
                    "sheet": str(sheet_name),
                    "row": int(row_number) + 2
                }
            )

            documents.append(document)

    return documents, ignored_columns


# =========================================================
# PDF PROCESSING
# =========================================================

def process_pdf(file_path):

    try:

        reader = PdfReader(
            str(file_path)
        )

    except Exception as e:

        raise Exception(
            f"Could not read PDF file: {e}"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    documents = []

    for page_number, page in enumerate(reader.pages):

        try:

            text = page.extract_text() or ""

        except Exception:

            text = ""

        if not text.strip():
            continue

        chunks = text_splitter.split_text(
            text
        )

        for chunk in chunks:

            document = Document(
                page_content=chunk,
                metadata={
                    "source": file_path.name,
                    "file_type": "PDF",
                    "page": page_number
                }
            )

            documents.append(document)

    return documents, []


# =========================================================
# TXT PROCESSING
# =========================================================

def process_txt(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

    except UnicodeDecodeError:

        try:

            with open(
                file_path,
                "r",
                encoding="latin-1"
            ) as f:

                text = f.read()

        except Exception as e:

            raise Exception(
                f"Could not read TXT file: {e}"
            )

    except Exception as e:

        raise Exception(
            f"Could not read TXT file: {e}"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(
        text
    )

    documents = []

    for chunk in chunks:

        document = Document(
            page_content=chunk,
            metadata={
                "source": file_path.name,
                "file_type": "TXT"
            }
        )

        documents.append(document)

    return documents, []


# =========================================================
# PROCESS ANY SUPPORTED FILE
# =========================================================

def process_file(file_path):

    extension = file_path.suffix.lower()

    if extension in [".xlsx", ".xls"]:

        return process_excel(file_path)

    elif extension == ".pdf":

        return process_pdf(file_path)

    elif extension == ".txt":

        return process_txt(file_path)

    else:

        return [], []


# =========================================================
# AUTOMATIC FILE INDEXING
# =========================================================

def index_new_files(progress_placeholder=None):

    vector_store = load_vector_store()

    indexed_files = load_indexed_files()

    supported_extensions = [
        ".pdf",
        ".txt",
        ".xlsx",
        ".xls"
    ]

    files = [
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower()
        in supported_extensions
    ]

    total_documents = 0

    ignored_columns = []

    indexed_count = 0

    for file_number, file_path in enumerate(files):

        if progress_placeholder:

            progress_placeholder.info(
                f"Processing: {file_path.name}"
            )

        try:

            current_hash = calculate_file_hash(
                file_path
            )

        except Exception:

            continue

        # Skip already indexed unchanged files
        if indexed_files.get(
            file_path.name
        ) == current_hash:

            continue

        try:

            documents, ignored = process_file(
                file_path
            )

        except Exception as e:

            st.error(
                f"Error processing "
                f"{file_path.name}: {e}"
            )

            continue

        if not documents:

            indexed_files[
                file_path.name
            ] = current_hash

            save_indexed_files(
                indexed_files
            )

            continue

        # Create unique IDs
        ids = []

        for index, document in enumerate(
            documents
        ):

            unique_string = (
                f"{file_path.name}|"
                f"{document.metadata.get('sheet', '')}|"
                f"{document.metadata.get('row', '')}|"
                f"{document.metadata.get('page', '')}|"
                f"{index}|"
                f"{current_hash}"
            )

            document_id = hashlib.md5(
                unique_string.encode("utf-8")
            ).hexdigest()

            ids.append(document_id)

        # Add documents to Chroma
        vector_store.add_documents(
            documents=documents,
            ids=ids
        )

        total_documents += len(documents)

        ignored_columns.extend(
            ignored
        )

        indexed_files[
            file_path.name
        ] = current_hash

        save_indexed_files(
            indexed_files
        )

        indexed_count += 1

    if progress_placeholder:

        progress_placeholder.empty()

    return (
        total_documents,
        indexed_count,
        ignored_columns
    )


# =========================================================
# BUILD BM25 INDEX
# =========================================================

@st.cache_resource
def build_bm25():

    vector_store = load_vector_store()

    collection = vector_store._collection

    result = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    all_docs = []

    documents = result.get(
        "documents",
        []
    )

    metadatas = result.get(
        "metadatas",
        []
    )

    for index, content in enumerate(
        documents
    ):

        metadata = {}

        if index < len(metadatas):

            metadata = (
                metadatas[index] or {}
            )

        all_docs.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    if not all_docs:

        return None, [], []

    tokenized_corpus = [

        re.findall(
            r"\b\w+\b",
            doc.page_content.lower()
        )

        for doc in all_docs
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return (
        bm25,
        all_docs,
        tokenized_corpus
    )


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(query, k=5):

    vector_store = load_vector_store()

    # -----------------------------------------------------
    # SEMANTIC SEARCH
    # -----------------------------------------------------

    semantic_results = (
        vector_store.similarity_search(
            query,
            k=k
        )
    )

    # -----------------------------------------------------
    # BM25 KEYWORD SEARCH
    # -----------------------------------------------------

    bm25, all_docs, tokenized_corpus = (
        build_bm25()
    )

    if bm25 is None:

        return semantic_results

    query_tokens = re.findall(
        r"\b\w+\b",
        query.lower()
    )

    if not query_tokens:

        return semantic_results

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    keyword_results = []

    for index in ranked_indices:

        if bm25_scores[index] <= 0:

            continue

        keyword_results.append(
            all_docs[index]
        )

        if len(keyword_results) >= k:

            break

    # -----------------------------------------------------
    # COMBINE RESULTS
    # -----------------------------------------------------

    combined_results = []

    seen = set()

    for document in (
        semantic_results +
        keyword_results
    ):

        key = (
            document.metadata.get(
                "source",
                ""
            ),

            document.metadata.get(
                "sheet",
                ""
            ),

            document.metadata.get(
                "row",
                ""
            ),

            document.metadata.get(
                "page",
                ""
            ),

            document.page_content[:200]
        )

        if key in seen:

            continue

        seen.add(key)

        combined_results.append(
            document
        )

        if len(combined_results) >= k:

            break

    return combined_results


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📚 Document Search"
)

st.sidebar.markdown(
    """
Upload your college documents.

Supported files:

- PDF
- TXT
- Excel (.xlsx)
- Excel (.xls)

Files are automatically indexed after upload.
"""
)


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_files = st.sidebar.file_uploader(
    "Upload College Documents",

    type=[
        "pdf",
        "txt",
        "xlsx",
        "xls"
    ],

    accept_multiple_files=True
)


# =========================================================
# AUTOMATIC UPLOAD + INDEXING
# =========================================================

if uploaded_files:

    files_saved = 0

    for uploaded_file in uploaded_files:

        destination = (
            DATA_DIR /
            uploaded_file.name
        )

        with open(
            destination,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        files_saved += 1

    st.sidebar.success(
        f"✅ {files_saved} file(s) uploaded."
    )

    progress = st.empty()

    with st.spinner(
        "Processing uploaded files..."
    ):

        try:

            (
                count,
                file_count,
                ignored
            ) = index_new_files(
                progress
            )

            # Important:
            # Chroma has changed, so BM25
            # must be rebuilt.

            build_bm25.clear()

            progress.empty()

            if count > 0:

                st.sidebar.success(
                    f"✅ Indexed {count} "
                    f"new records/chunks."
                )

            else:

                st.sidebar.info(
                    "Files are already indexed."
                )

            if ignored:

                st.sidebar.warning(
                    "🔐 Sensitive Excel "
                    "columns ignored: "
                    + str(len(ignored))
                )

        except Exception as e:

            progress.empty()

            st.error(
                f"Automatic indexing error: {e}"
            )


# =========================================================
# MAIN PAGE
# =========================================================

st.title(
    "📚 Information Retrieval and "
    "Hybrid RAG-Based Document "
    "Question Answering System"
)

st.write(
    "Upload documents and ask questions "
    "about their content."
)


# =========================================================
# USER QUERY
# =========================================================

st.subheader(
    "🔎 Ask a Question"
)

user_query = st.text_input(
    "Enter your question:",

    placeholder=(
        "Example: What is a binomial heap?"
    )
)


# =========================================================
# SEARCH BUTTON
# =========================================================

if st.button(
    "🔎 Search Knowledge Base",
    type="primary"
):

    if not user_query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            # Check whether database contains data
            vector_store = load_vector_store()

            collection = (
                vector_store._collection
            )

            collection_data = (
                collection.get(
                    include=["documents"]
                )
            )

            documents = (
                collection_data.get(
                    "documents",
                    []
                )
            )

            if not documents:

                st.warning(
                    "No documents are indexed. "
                    "Please upload a file first."
                )

            else:

                # -------------------------------------------------
                # SEARCH
                # -------------------------------------------------

                results = hybrid_search(
                    user_query,
                    k=5
                )

                if not results:

                    st.warning(
                        "No relevant information found."
                    )

                else:

                    # -------------------------------------------------
                    # BEST MATCH
                    # -------------------------------------------------

                    best = results[0]

                    st.success(
                        "### Best Matching Answer"
                    )

                    st.write(
                        best.page_content
                    )

                    # -------------------------------------------------
                    # SOURCE
                    # -------------------------------------------------

                    source = best.metadata.get(
                        "source",
                        "Unknown"
                    )

                    file_type = best.metadata.get(
                        "file_type",
                        ""
                    )

                    source_text = (
                        f"📄 **Source:** {source}"
                    )

                    if file_type:

                        source_text += (
                            f" | **Type:** {file_type}"
                        )

                    if best.metadata.get(
                        "sheet"
                    ):

                        source_text += (
                            f" | **Sheet:** "
                            f"{best.metadata.get('sheet')}"
                        )

                    if best.metadata.get(
                        "row"
                    ):

                        source_text += (
                            f" | **Row:** "
                            f"{best.metadata.get('row')}"
                        )

                    if best.metadata.get(
                        "page"
                    ) is not None:

                        source_text += (
                            f" | **Page:** "
                            f"{int(best.metadata.get('page')) + 1}"
                        )

                    st.caption(
                        source_text
                    )

                    # -------------------------------------------------
                    # OTHER MATCHING RESULTS
                    # -------------------------------------------------

                    if len(results) > 1:

                        st.subheader(
                            "📑 Other Relevant Results"
                        )

                        for i, document in enumerate(
                            results[1:],
                            start=2
                        ):

                            with st.expander(
                                f"Result {i} — "
                                f"{document.metadata.get('source', 'Unknown')}"
                            ):

                                st.write(
                                    document.page_content
                                )

                                result_source = (
                                    document.metadata.get(
                                        "source",
                                        "Unknown"
                                    )
                                )

                                st.caption(
                                    f"Source: "
                                    f"{result_source}"
                                )

        except Exception as e:

            st.error(
                f"Search error: {e}"
            )


# =========================================================
# SIDEBAR INFORMATION
# =========================================================

st.sidebar.markdown("---")

st.sidebar.info(
    """
### 🔍 Retrieval

This system uses:

• ChromaDB  
• HuggingFace Embeddings  
• BM25  
• Hybrid Retrieval  

### 📄 Supported Files

• PDF  
• TXT  
• XLSX  
• XLS  

### 🔐 Security

Sensitive Excel columns such as:

• Password  
• Secret  
• PWD  

are automatically ignored.
"""
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Hybrid Retrieval using ChromaDB + BM25"
)