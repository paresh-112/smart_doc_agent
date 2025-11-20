"""Vector database operations using ChromaDB."""
import os
from typing import List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from utils.logger import get_logger

logger = get_logger('vectordb')


class VectorStore:
    """Manage vector storage and retrieval using ChromaDB."""

    def __init__(self, api_key: str, session_id: str, collection_name: str = None):
        """Initialize vector store.

        Args:
            api_key: Google API key for embeddings
            session_id: Unique session identifier for isolation
            collection_name: Optional name of the collection in ChromaDB

        Note:
            Each session uses its own ChromaDB collection to ensure data isolation.
            This prevents chunks from different sessions/documents from mixing.
        """
        logger.info(f"Initializing VectorStore for session: {session_id[:8]}...")

        try:
            self.api_key = api_key
            # Use session-specific collection name for data isolation
            # This ensures each session has its own isolated vector store
            self.collection_name = collection_name or f"session_{session_id}"
            self.session_id = session_id

            # Initialize Google Generative AI embeddings
            # Using text-embedding-004 which is the latest stable embedding model
            logger.debug("Initializing Google embeddings model")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )

            # Detect if running on Streamlit Cloud (read-only filesystem)
            # Check for Streamlit Cloud environment
            is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src')

            if is_streamlit_cloud:
                # Use in-memory storage for Streamlit Cloud
                logger.info("Detected Streamlit Cloud - using in-memory ChromaDB")
                self.persist_directory = None
            else:
                # Use persistent storage for local deployment
                self.persist_directory = "./chroma_db"
                if not os.path.exists(self.persist_directory):
                    logger.info(f"Creating ChromaDB directory: {self.persist_directory}")
                    os.makedirs(self.persist_directory, exist_ok=True)

            # Initialize ChromaDB client
            self.vectorstore = None
            logger.info("✅ VectorStore initialized successfully")

        except Exception as e:
            logger.error(f"❌ VectorStore initialization failed: {str(e)}", exc_info=True)
            raise

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create or update vector store with documents.

        Args:
            documents: List of Document objects to store
        """
        logger.info(f"Creating vectorstore with {len(documents)} documents")

        try:
            # Validate documents
            if not documents:
                raise ValueError("No documents provided to store")

            # Filter out any documents with empty content
            valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]

            if not valid_documents:
                raise ValueError("All documents have empty content")

            logger.info(f"Processing {len(valid_documents)} valid documents")

            if self.vectorstore is None:
                # Create new vectorstore
                logger.info(f"Creating new vectorstore collection: {self.collection_name}")

                if self.persist_directory:
                    # Persistent storage (local)
                    self.vectorstore = Chroma.from_documents(
                        documents=valid_documents,
                        embedding=self.embeddings,
                        collection_name=self.collection_name,
                        persist_directory=self.persist_directory
                    )
                else:
                    # In-memory storage (Streamlit Cloud)
                    self.vectorstore = Chroma.from_documents(
                        documents=valid_documents,
                        embedding=self.embeddings,
                        collection_name=self.collection_name
                    )
                logger.info("✅ Vectorstore created successfully")
            else:
                # Add documents to existing vectorstore
                logger.info("Adding documents to existing vectorstore")
                self.vectorstore.add_documents(valid_documents)
                logger.info("✅ Documents added to vectorstore")

        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {str(e)}", exc_info=True)
            raise

    def load_vectorstore(self) -> bool:
        """Load existing vector store.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if self.persist_directory:
                # Load from persistent storage (local)
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                return True
            else:
                # In-memory storage - cannot reload (Streamlit Cloud)
                # Vectorstore only exists in current session
                logger.debug("In-memory vectorstore - cannot reload")
                return False
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            return False

    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        logger.debug(f"Searching for similar documents (k={k}): {query[:50]}...")

        try:
            if self.vectorstore is None:
                self.load_vectorstore()

            if self.vectorstore is None:
                logger.warning("No vectorstore available for search")
                return []

            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching vectorstore: {str(e)}", exc_info=True)
            return []

    def search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of tuples (document, score)
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore is None:
            return []

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def clear_vectorstore(self) -> None:
        """Clear the vector store."""
        try:
            if self.vectorstore is not None:
                logger.info(f"Clearing vectorstore collection: {self.collection_name}")
                self.vectorstore.delete_collection()
                self.vectorstore = None
                logger.info("✅ Vectorstore cleared successfully")
            else:
                logger.debug("No vectorstore to clear")
        except Exception as e:
            logger.error(f"Error clearing vectorstore: {str(e)}", exc_info=True)

    def get_retriever(self, k: int = 4):
        """Get a retriever object for the vectorstore.

        Args:
            k: Number of documents to retrieve

        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore is None:
            return None

        return self.vectorstore.as_retriever(search_kwargs={"k": k})
