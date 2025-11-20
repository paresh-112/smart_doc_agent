
"""Smart Document Agent - Chat with your documents using Gemini AI."""
import os
import uuid
import shutil
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from audio_recorder_streamlit import audio_recorder

from utils.document_processor import DocumentProcessor
from utils.chunking import SmartChunker
from utils.vectordb import VectorStore
from utils.database import SessionDatabase
from utils.voice_handler import VoiceHandler
from utils.logger import get_logger
from utils.config import get_config, ConfigError

# Initialize logger
logger = get_logger('app')

# Load environment variables
load_dotenv()

# Initialize and validate configuration
try:
    config = get_config()
    logger.info("=" * 80)
    logger.info("SMART DOCUMENT AGENT - PRODUCTION MODE")
    logger.info("=" * 80)
except ConfigError as e:
    logger.error(f"Configuration error: {str(e)}")
    st.error(f"⚠️ Configuration Error: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Document Agent",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New browser session started: {st.session_state.session_id[:8]}...")

if 'db' not in st.session_state:
    try:
        logger.info("Initializing database connection")
        st.session_state.db = SessionDatabase(config.database_path)

        # Clear all old sessions on browser refresh/initial load
        logger.info("Clearing previous sessions (fresh start)")
        st.session_state.db.clear_all_sessions()

        # Clear ChromaDB on fresh start (only for local, not Streamlit Cloud)
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src')

        if not is_streamlit_cloud and config.chroma_db_path:
            chroma_dir = config.chroma_db_path
            if os.path.exists(chroma_dir):
                try:
                    logger.info(f"Clearing ChromaDB directory: {chroma_dir}")
                    shutil.rmtree(chroma_dir)
                    os.makedirs(chroma_dir, exist_ok=True)
                    logger.info("✅ ChromaDB cleared")
                except Exception as e:
                    logger.error(f"Failed to clear ChromaDB: {str(e)}")
        else:
            logger.info("Using in-memory ChromaDB storage")

        # Create new session in database
        st.session_state.db.create_session(st.session_state.session_id)
        logger.info("✅ Session initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize session: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = []
if 'image_data' not in st.session_state:
    st.session_state.image_data = None
if 'current_doc_data' not in st.session_state:
    st.session_state.current_doc_data = None
if 'message_sources' not in st.session_state:
    st.session_state.message_sources = []
if 'voice_handler' not in st.session_state:
    st.session_state.voice_handler = VoiceHandler()
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = 'en'
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None
if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False


def initialize_components(api_key: str, session_id: str):
    """Initialize document processor, chunker, and vector store."""
    doc_processor = DocumentProcessor()
    chunker = SmartChunker(chunk_size=1000, chunk_overlap=200)
    vectorstore = VectorStore(api_key=api_key, session_id=session_id)

    return doc_processor, chunker, vectorstore


def load_session(session_id: str, chunker, api_key: str):
    """Load a previous session from database.

    Args:
        session_id: Session ID to load
        chunker: Chunker instance
        api_key: Google API key for creating new vectorstore
    """
    # Clear current session
    if st.session_state.vectorstore:
        st.session_state.vectorstore.clear_vectorstore()

    # Update session ID
    st.session_state.session_id = session_id
    st.session_state.vectorstore = None
    st.session_state.documents_processed = []
    st.session_state.image_data = None
    st.session_state.current_doc_data = None
    st.session_state.message_sources = []

    # Load messages
    messages = st.session_state.db.get_session_messages(session_id)
    st.session_state.messages = messages

    # Load document
    doc_info = st.session_state.db.get_session_document(session_id)
    if doc_info and doc_info['doc_data']:
        doc_data = doc_info['doc_data']
        st.session_state.current_doc_data = doc_data
        st.session_state.documents_processed = [doc_info['filename']]

        # Restore image data if it's an image
        if doc_data['type'] == 'image':
            st.session_state.image_data = doc_data
        elif doc_info['chunks_count'] > 0:
            # Re-chunk and restore vector store for text documents with session-specific collection
            new_vectorstore = VectorStore(api_key=api_key, session_id=session_id)
            chunks = chunker.chunk_document(doc_data)
            if chunks:
                new_vectorstore.create_vectorstore(chunks)
                st.session_state.vectorstore = new_vectorstore

    st.rerun()


def create_new_session():
    """Create a new session."""
    # Clear current session
    if st.session_state.vectorstore:
        st.session_state.vectorstore.clear_vectorstore()

    # Create new session
    new_session_id = str(uuid.uuid4())
    st.session_state.db.create_session(new_session_id)

    # Reset state
    st.session_state.session_id = new_session_id
    st.session_state.vectorstore = None
    st.session_state.documents_processed = []
    st.session_state.messages = []
    st.session_state.message_sources = []
    st.session_state.image_data = None
    st.session_state.current_doc_data = None

    st.rerun()


def process_and_store_document(uploaded_file, doc_processor, chunker, vectorstore):
    """Process uploaded document and store in vector database."""
    try:
        logger.info(f"Processing document: {uploaded_file.name} ({uploaded_file.size} bytes)")

        # Validate file size
        max_size_mb = config.max_file_size_mb
        if not config.validate_file_size(uploaded_file.size, max_size_mb):
            error_msg = f"File size ({uploaded_file.size / (1024*1024):.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
            logger.warning(error_msg)
            st.error(f"❌ {error_msg}")
            return False

        # Clear previous documents (only one document at a time)
        if st.session_state.vectorstore:
            logger.info("Clearing previous vectorstore")
            st.session_state.vectorstore.clear_vectorstore()
        st.session_state.vectorstore = None
        st.session_state.documents_processed = []
        st.session_state.image_data = None
        st.session_state.current_doc_data = None

        # Process the document
        logger.info(f"Processing file: {uploaded_file.name}")
        doc_data = doc_processor.process_file(uploaded_file)

        # Store document data for viewing
        st.session_state.current_doc_data = doc_data

        # Store image data separately for vision capabilities
        if doc_data['type'] == 'image':
            st.session_state.image_data = doc_data
            st.session_state.documents_processed.append(uploaded_file.name)

            # Save to database
            st.session_state.db.save_document(
                st.session_state.session_id,
                uploaded_file.name,
                doc_data['type'],
                doc_data,
                0  # No chunks for images
            )

            # Update session title with document name
            st.session_state.db.update_session_title(
                st.session_state.session_id,
                f"📷 {uploaded_file.name[:30]}"
            )

            st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
            return True

        # Chunk the document
        chunks = chunker.chunk_document(doc_data)

        # Check if we have valid chunks
        if not chunks or len(chunks) == 0:
            st.error(f"❌ No valid content found in document '{uploaded_file.name}'. The document might be empty or contain only images/formatting.")
            return False

        # Store in vector database
        vectorstore.create_vectorstore(chunks)
        st.session_state.vectorstore = vectorstore

        # Add to processed documents
        st.session_state.documents_processed.append(uploaded_file.name)

        # Save to database
        st.session_state.db.save_document(
            st.session_state.session_id,
            uploaded_file.name,
            doc_data['type'],
            doc_data,
            len(chunks)
        )

        # Update session title with document name
        doc_icon = "📄" if doc_data['type'] == 'pdf' else "📝"
        st.session_state.db.update_session_title(
            st.session_state.session_id,
            f"{doc_icon} {uploaded_file.name[:30]}"
        )

        st.success(f"✅ Document '{uploaded_file.name}' processed successfully! ({len(chunks)} chunks created)")
        logger.info(f"✅ Document processed: {uploaded_file.name} ({len(chunks)} chunks)")
        return True

    except Exception as e:
        logger.error(f"❌ Error processing document {uploaded_file.name}: {str(e)}", exc_info=True)
        st.error(f"❌ Error processing document: {str(e)}")
        return False


def answer_question_with_vision(question: str, image_path: str, api_key: str):
    """Answer questions about images using Gemini vision capabilities with streaming."""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Load the image
        img = Image.open(image_path)

        # Use Gemini 2.5 Flash with vision
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Create prompt with security and instruction guidelines
        prompt = f"""You are an AI assistant trained by the Smart Document Agent team to help users with their documents.

IMPORTANT INSTRUCTIONS:
1. GREETING CHECK: If the user's message is a greeting (hello, hi, hey, namaste, etc.), respond warmly with a greeting and ask: "How can I help you with your document today?" in their language. Do NOT analyze the image for greetings.

2. LANGUAGE: Detect the language of the user's question and respond in the SAME language (English, Gujarati, Hindi, etc.)

3. Answer the question based ONLY on what you can see in this image
4. If you cannot see or determine the answer from the image, say "I don't know" or "I cannot determine this from the image" in the user's language
5. Do NOT make up information that is not visible in the image
6. Do NOT reveal your model name, version, or technical details
7. If asked about your identity, say: "I am an AI assistant trained by the Smart Document Agent team to help you with your documents" in the user's language
8. Do NOT answer personal questions about yourself
9. Focus exclusively on analyzing and describing this image/document

User Question: {question}

Answer based on the image (respond in the same language as the question):"""

        # Generate streaming response
        response = model.generate_content([prompt, img], stream=True)

        # Yield chunks for streaming
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"Error processing image question: {str(e)}"


def answer_question_with_docs(question: str, vectorstore: VectorStore, api_key: str):
    """Smart RAG with adaptive query rewriting and confidence-based decisions."""
    MAX_ATTEMPTS = 3

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    current_query = question
    original_question = question

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Step 1: Retrieve documents with k=6 for better coverage
            docs = vectorstore.search_similar(current_query, k=6)

            if not docs or len(docs) == 0:
                if attempt < MAX_ATTEMPTS:
                    # No docs found, simple rewrite
                    current_query = f"Find information about: {current_query}"
                    continue
                else:
                    yield f"I don't know - I couldn't find relevant information in the document after {MAX_ATTEMPTS} attempts."
                    yield []
                    return

            # Step 2: Build context from retrieved documents
            context = "\n\n".join([f"[Chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])

            # Step 3: LLM with confidence-based decision
            confidence_prompt = f"""You are an AI assistant trained by the Smart Document Agent team to help users with their documents.

CONTEXT FROM DOCUMENT:
{context}

ORIGINAL USER QUESTION: {original_question}

CURRENT SEARCH QUERY: {current_query}

ATTEMPT: {attempt} of {MAX_ATTEMPTS}

INSTRUCTIONS:
1. GREETING CHECK: If the ORIGINAL USER QUESTION is a greeting (hello, hi, hey, namaste, good morning, etc.), respond warmly with a greeting and ask: "How can I help you with your document today?" in their language. IGNORE the context for greetings.

2. LANGUAGE: Detect the language of the ORIGINAL USER QUESTION and respond in the SAME language (English, Gujarati, Hindi, etc.)

3. READ the context carefully

4. Decide if you can answer the ORIGINAL question with HIGH CONFIDENCE based on the context

5. If you CAN answer confidently (info is clearly in the context):
   - Provide a clear, direct answer in the user's language
   - If the question has multiple parts, answer each one
   - NEVER make up information not in the context
   - Do NOT reveal your model name, version, or technical details
   - If asked about your identity, say: "I am an AI assistant trained by the Smart Document Agent team" in the user's language

6. If you CANNOT answer confidently (info not in context or unclear):
   - Respond EXACTLY in this format: "REWRITE: [better phrased question]"
   - Make the rewrite more specific, use synonyms, or break down the question
   - Examples:
     * "What's the amount?" → "REWRITE: What is the total amount, sum, or payment value?"
     * "Who sent this?" → "REWRITE: Who is the sender, issuer, or author?"

SECURITY RULES:
- Do NOT answer personal questions about yourself
- Do NOT engage in topics unrelated to the document
- Focus exclusively on helping with document-related queries

YOUR RESPONSE (in the same language as the user's question):"""

            # Generate response with streaming
            full_response = ""
            response = model.generate_content(confidence_prompt, stream=True)

            # Collect response to check for REWRITE
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text

            # Step 4: Check if LLM requested a rewrite
            if full_response.strip().startswith("REWRITE:"):
                if attempt < MAX_ATTEMPTS:
                    # Extract the rewritten query
                    rewritten = full_response.replace("REWRITE:", "").strip()
                    current_query = rewritten
                    # Log for debugging
                    yield f"🔄 Refining search... (Attempt {attempt + 1}/{MAX_ATTEMPTS})\n\n"
                    continue
                else:
                    # Max attempts reached, still not confident
                    yield f"I don't know - I couldn't find sufficient information in the document after {MAX_ATTEMPTS} attempts. The information you're looking for may not be in this document."
                    yield []
                    return

            else:
                # LLM is confident, stream the answer character by character
                for char in full_response:
                    yield char

                # After streaming complete, yield sources
                yield docs
                return

        except Exception as e:
            if attempt < MAX_ATTEMPTS:
                continue
            else:
                yield f"Error answering question: {str(e)}"
                yield []
                return

    # Safety net (should not reach here)
    yield "I don't know - unable to process your question."
    yield []
    return


def main():
    """Main application."""
    # Brand logo in upper left corner
    st.markdown("""
    <style>
    .brand-header {
        display: flex;
        align-items: center;
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 1rem;
    }
    .brand-logo {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    .brand-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
    }
    </style>
    <div class="brand-header">
        <span class="brand-logo">📚</span>
        <span class="brand-name">Smart Document Agent</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for document upload
    with st.sidebar:
        # Get API Key from environment
        api_key = config.google_api_key

        if not api_key:
            logger.error("Google API key not found in environment")
            st.error("⚠️ Please set GOOGLE_API_KEY in .env file")
            st.stop()

        logger.debug("API key loaded successfully")

        # Initialize components
        doc_processor, chunker, vectorstore = initialize_components(api_key, st.session_state.session_id)

        # Drag and drop file uploader with custom styling
        st.markdown("""
        <style>
        [data-testid="stFileUploader"] {
            padding: 2rem 1rem;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.05);
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(255, 75, 75, 0.6);
            background-color: rgba(255, 255, 255, 0.08);
        }
        </style>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📎 Drag & Drop or Browse",
            type=['pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg'],
            label_visibility="visible"
        )

        if uploaded_file:
            if st.button("✓ Process Document", width='stretch'):
                with st.spinner("Processing document..."):
                    process_and_store_document(uploaded_file, doc_processor, chunker, vectorstore)

        # Show current document
        if st.session_state.documents_processed:
            st.success(f"✓ {st.session_state.documents_processed[0]}")

            # Clear button
            if st.button("🗑️ Clear Document", width='stretch'):
                if st.session_state.vectorstore:
                    st.session_state.vectorstore.clear_vectorstore()
                st.session_state.vectorstore = None
                st.session_state.documents_processed = []
                st.session_state.messages = []
                st.session_state.message_sources = []
                st.session_state.image_data = None
                st.session_state.current_doc_data = None
                st.session_state.db.clear_session_messages(st.session_state.session_id)
                st.rerun()

        # Session Management
        st.divider()

        # New session and clear all buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Session", width='stretch'):
                create_new_session()
        with col2:
            if st.button("🗑️ Clear All", width='stretch'):
                # Clear current vector store
                if st.session_state.vectorstore:
                    st.session_state.vectorstore.clear_vectorstore()

                # Clear all ChromaDB collections (only if using persistent storage)
                if config.chroma_db_path:
                    chroma_dir = config.chroma_db_path
                    if os.path.exists(chroma_dir):
                        try:
                            shutil.rmtree(chroma_dir)
                            os.makedirs(chroma_dir, exist_ok=True)
                        except Exception as e:
                            st.warning(f"⚠️ Could not clear vector database: {e}")

                # Clear all sessions from database
                if st.session_state.db.clear_all_sessions():
                    # Create a new session
                    create_new_session()
                    st.success("✅ All sessions cleared!")
                else:
                    st.error("❌ Failed to clear sessions")

        # Get all sessions
        sessions = st.session_state.db.get_all_sessions()

        if sessions:
            st.write(f"**Sessions ({len(sessions)})**")

            # Display sessions
            for session in sessions[:10]:  # Show last 10 sessions
                is_current = session['session_id'] == st.session_state.session_id

                # Create session item with delete button
                session_col1, session_col2 = st.columns([4, 1])

                with session_col1:
                    # Show session with indicator if current
                    label = f"{'🟢 ' if is_current else ''}{session['title']}"
                    if st.button(
                        label,
                        key=f"session_{session['session_id']}",
                        disabled=is_current,
                        width='stretch'
                    ):
                        load_session(session['session_id'], chunker, api_key)

                with session_col2:
                    if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete session"):
                        st.session_state.db.delete_session(session['session_id'])
                        if is_current:
                            create_new_session()
                        else:
                            st.rerun()

                # Show session info
                with st.expander(f"ℹ️ Details", expanded=False):
                    st.caption(f"📄 Documents: {session['doc_count']}")
                    st.caption(f"💬 Messages: {session['msg_count']}")
                    st.caption(f"🕐 Updated: {session['updated_at']}")
        else:
            st.info("No sessions yet")

    # Main interface with tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📄 Document Viewer"])

    # Chat Tab
    with tab1:
        # Create a container for chat messages with custom styling
        st.markdown("""
        <style>
        /* Chat container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 10rem;
            max-width: 900px;
        }

        /* Chat message styling */
        .stChatMessage {
            margin-bottom: 1.5rem;
            padding: 1rem;
            border-radius: 12px;
        }

        /* User message styling */
        [data-testid="stChatMessageContent"] {
            background-color: transparent;
        }

        /* Input section at bottom */
        .stChatInputContainer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1.5rem 2rem 1.5rem 340px;
            background: linear-gradient(to top, rgba(14, 17, 23, 0.98) 80%, transparent);
            backdrop-filter: blur(12px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 999;
        }

        /* Adjust for sidebar */
        section[data-testid="stSidebar"] ~ .main .stChatInputContainer {
            padding-left: 340px;
        }

        /* Chat input box */
        .stChatInput > div {
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }

        /* Source expander styling */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)

        # Display chat messages in a scrollable container
        chat_container = st.container()
        with chat_container:
            if st.session_state.messages:
                assistant_msg_count = 0  # Track assistant messages separately
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                        # Show sources for assistant messages if available
                        if message["role"] == "assistant":
                            if assistant_msg_count < len(st.session_state.get('message_sources', [])):
                                sources = st.session_state.message_sources[assistant_msg_count]
                                if sources:
                                    with st.expander("📚 View Sources", expanded=False):
                                        for s_idx, doc in enumerate(sources):
                                            st.markdown(f"**Source {s_idx + 1}:**")
                                            st.markdown(f"- File: {doc.metadata.get('source', 'Unknown')}")
                                            if 'page' in doc.metadata:
                                                st.markdown(f"- Page: {doc.metadata['page']}")
                                            st.markdown(f"```\n{doc.page_content[:200]}...\n```")
                                            if s_idx < len(sources) - 1:
                                                st.divider()
                            assistant_msg_count += 1  # Increment for next assistant message

        # Input section at the bottom - ChatGPT style
        # Create columns for text input and voice button
        input_col, voice_col = st.columns([10, 1])

        with input_col:
            # Text input
            text_question = st.chat_input("Ask a question about your document...")

        with voice_col:
            # Voice recorder button (compact)
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="1x",
            )

        # Process voice input if audio is recorded (only if it's new audio)
        question = None
        if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
            # Mark this audio as processed to prevent re-processing
            st.session_state.last_audio_bytes = audio_bytes
            st.session_state.processing_audio = True

            with st.spinner("🎤 Transcribing..."):
                text, lang_code = st.session_state.voice_handler.transcribe_audio(audio_bytes)
                if text:
                    question = text
                    st.session_state.detected_language = lang_code
                else:
                    st.error("❌ Could not transcribe. Please try again.")
                    st.session_state.processing_audio = False
        elif text_question:
            # Use text input if provided
            question = text_question
            st.session_state.detected_language = st.session_state.voice_handler.detect_language_from_text(question)

        if question:
            # Check if any documents are processed
            if not st.session_state.documents_processed:
                st.warning("⚠️ Please upload and process a document first!")
                st.stop()

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.db.save_message(st.session_state.session_id, "user", question)

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(question)

            # Generate and stream response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                sources = []

                # Check if we have an image or text documents
                if st.session_state.image_data:
                    # Use vision capabilities with streaming
                    for chunk in answer_question_with_vision(
                        question,
                        st.session_state.image_data['temp_path'],
                        api_key
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                elif st.session_state.vectorstore:
                    # Use document retrieval with adaptive smart RAG
                    stream_generator = answer_question_with_docs(
                        question,
                        st.session_state.vectorstore,
                        api_key
                    )

                    # Stream the response
                    for item in stream_generator:
                        if isinstance(item, str):
                            # This is a text chunk (character or message)
                            full_response += item
                            response_placeholder.markdown(full_response + "▌")
                        elif isinstance(item, list):
                            # This is the sources list (returned at the end)
                            sources = item

                else:
                    full_response = "No documents available. Please upload a document first."
                    response_placeholder.markdown(full_response)

                # Remove cursor and show final response
                response_placeholder.markdown(full_response)

                # Show sources if available (below the answer)
                if sources and len(sources) > 0:
                    st.markdown("---")
                    with st.expander(f"📚 View Sources ({len(sources)} chunks)", expanded=False):
                        for s_idx, doc in enumerate(sources):
                            st.markdown(f"**Source {s_idx + 1}:**")
                            st.markdown(f"- **File:** {doc.metadata.get('source', 'Unknown')}")
                            if 'page' in doc.metadata:
                                st.markdown(f"- **Page:** {doc.metadata['page']}")
                            st.markdown(f"**Content:**")
                            st.code(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            if s_idx < len(sources) - 1:
                                st.divider()

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.message_sources.append(sources)
            st.session_state.db.save_message(st.session_state.session_id, "assistant", full_response)

            # Reset processing flag to prepare for next input
            # NOTE: Don't reset last_audio_bytes here - keep it to prevent re-processing same audio
            st.session_state.processing_audio = False

            # Rerun to display the new messages
            st.rerun()

    # Document Viewer Tab
    with tab2:
        if not st.session_state.current_doc_data:
            st.info("📝 No document loaded. Please upload and process a document first.")
        else:
            doc_data = st.session_state.current_doc_data
            st.subheader(f"📄 {doc_data['filename']}")

            # Display based on document type
            if doc_data['type'] == 'pdf':
                st.write(f"**Total Pages:** {doc_data['total_pages']}")
                st.divider()

                # Display each page
                for page_data in doc_data['pages']:
                    with st.expander(f"Page {page_data['page']}", expanded=(page_data['page'] == 1)):
                        st.text_area(
                            f"Content of Page {page_data['page']}",
                            value=page_data['content'],
                            height=300,
                            key=f"page_{page_data['page']}",
                            disabled=True
                        )

            elif doc_data['type'] == 'docx':
                st.write(f"**Total Paragraphs:** {doc_data['total_paragraphs']}")
                st.divider()

                # Display full content
                full_content = "\n\n".join([p['content'] for p in doc_data['paragraphs']])
                st.text_area(
                    "Document Content",
                    value=full_content,
                    height=500,
                    disabled=True
                )

            elif doc_data['type'] == 'image':
                st.write(f"**Size:** {doc_data['size'][0]} x {doc_data['size'][1]} pixels")
                st.write(f"**Format:** {doc_data['format']}")
                st.divider()

                # Display the image
                img = Image.open(doc_data['temp_path'])
                st.image(img, caption=doc_data['filename'], use_container_width=True)


if __name__ == "__main__":
    main()
