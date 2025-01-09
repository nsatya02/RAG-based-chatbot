import os
import hashlib
import progressbar
import uuid
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import io
import tempfile

# Database configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "beamx",
    "host": "localhost",
    "port": "5432"
}

# Connection pool
CONNECTION_POOL = SimpleConnectionPool(1, 10, **DB_CONFIG)

qdrant_client = QdrantClient(
    url="https://f6a53cca-c940-4be3-a755-2e487985c694.europe-west3-0.gcp.cloud.qdrant.io:6333/", 
    api_key="tKa_u_ijF7p85Y7pDaoMbtIBhx9ZpXbdCOm1wH1BNQMDgh1j_zkECg",
)

COLLECTION_NAME = "pdf_documents"

qdrant_client.set_model("sentence-transformers/all-MiniLM-L6-v2")
qdrant_client.set_sparse_model("prithivida/Splade_PP_en_v1")

def get_db_connection():
    """Get a connection from the pool."""
    try:
        conn = CONNECTION_POOL.getconn()
        #print("Database connection established.")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

def release_db_connection(conn):
    """Release the connection back to the pool."""
    CONNECTION_POOL.putconn(conn)

def ensure_table_exists():
    """Ensure the documents table exists in the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    added_to_vectorstore BOOLEAN DEFAULT FALSE
                )
                """
            )
        conn.commit()
    except Exception as e:
        print(f"Error ensuring table exists: {e}")
        conn.rollback()
    finally:
        release_db_connection(conn)

def is_duplicate_file(conn, file_hash):
    """Check if the file hash already exists in the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE file_hash = %s", (file_hash,))
        return cur.fetchone() is not None

def add_file_to_db(conn, file_name, file_hash):
    """Add file metadata to the database if it is not a duplicate."""
    try:
        # Check if the file is a duplicate
        if is_duplicate_file(conn, file_hash):
            st.sidebar.success(f"Duplicate file detected: {file_name}. Skipping insertion.")
            return  # Skip adding if it's a duplicate

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (file_name, file_hash) VALUES (%s, %s)",
                (file_name, file_hash)
            )
        conn.commit()  # Ensure the transaction is committed
        st.sidebar.success(f"Successfully added {file_name} to the database.")
        st.sidebar.success("Files processed and added to the Qdrant vectorstore.")
    except Exception as e:
        print(f"Error inserting file metadata: {e}")
        conn.rollback()  # Rollback in case of error

def add_success_status_to_db(conn, file_name):
    """Update the file status to indicate that it was successfully added to the vectorstore."""
    try:
        #print(f"Updating status for file: {file_name}")  # Log the file name
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET added_to_vectorstore = TRUE WHERE file_name = %s",
                (file_name,)
            )
        conn.commit()  # Ensure the transaction is committed
        #print(f"Successfully updated status for {file_name} in PostgreSQL.")
    except Exception as e:
        print(f"Error updating success status for {file_name}: {e}")
        conn.rollback()  # Rollback in case of error

def generate_pdf_hash(file):
    """Generate a unique hash for a PDF file based on its content."""
    hasher = hashlib.sha256()
    while chunk := file.read(8192):
        hasher.update(chunk)
    return hasher.hexdigest()

def process_uploaded_files(uploaded_files):
    """Process files uploaded via Streamlit and return the document content."""
    raw_pdf_elements = {}

    # Create a temporary directory to store the files
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Save each uploaded file to the temporary directory
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Write the uploaded file to the temporary file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Get the file content from the uploaded file buffer

            try:
                # Open the file in binary mode for hashing
                with open(file_path, "rb") as f:
                    pdf_hash = generate_pdf_hash(f)  # Pass the file object to the hash function

                # Now pass the file path to PyPDFLoader
                loader = PyPDFLoader(file_path)  # Should be a valid file path here
                documents = loader.load()

                # Split the documents using CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=4000,
                    chunk_overlap=200
                )
                raw_pdf_elements[pdf_hash] = splitter.split_documents(documents)

            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {e}")

    return raw_pdf_elements

def initialize_vectorstore(uploaded_files):
    """Initialize the Qdrant vectorstore with uploaded PDF data."""
    embeddings = HuggingFaceEmbeddings()

    # Ensure the documents table exists
    ensure_table_exists()

    # Process PDFs and get elements
    raw_pdf_elements = process_uploaded_files(uploaded_files)
    if not raw_pdf_elements:
        print("No new PDFs to process.")
        return

    # Combine all documents into a single list
    all_documents = []
    for docs in raw_pdf_elements.values():
        all_documents.extend(docs)

    # Check if the Qdrant collection exists
    collections = qdrant_client.get_collections()
    if COLLECTION_NAME not in [col.name for col in collections.collections]:
        # Create collection if it doesn't exist
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,  # Adjust accordingly
                distance=Distance.COSINE
            ),
        )
        print(f"Collection {COLLECTION_NAME} created.")
    else:
        print(f"Collection {COLLECTION_NAME} already exists. Skipping creation.")

    # Add documents to Qdrant
    conn = get_db_connection()
    try:
        for i, doc in enumerate(all_documents):
            vector = embeddings.embed_query(doc.page_content)
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[{
                    "id": str(uuid.uuid4()),  # Generates a unique UUID string
                    "vector": vector,
                    "payload": {"text": doc.page_content}
                }]
            )
        #print(f"Documents added to Qdrant collection: {COLLECTION_NAME}")

        # After successfully adding the document to the vectorstore, update the status
        for uploaded_file in uploaded_files:
            file_hash = generate_pdf_hash(uploaded_file)  # Make sure you generate the hash
            add_file_to_db(conn, uploaded_file.name, file_hash)  # Insert metadata if not duplicate
            add_success_status_to_db(conn, uploaded_file.name)  # Mark as added to vectorstore
            #print(f"Updated success status for file: {uploaded_file.name}")

    except Exception as e:
        print(f"Error adding documents to Qdrant: {e}")
    finally:
        release_db_connection(conn)





