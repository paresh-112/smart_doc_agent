# 📚 Smart Document Agent

A powerful AI-powered document chat application that allows you to upload documents (PDF, DOC, DOCX, Images) and have intelligent conversations with them using Google's Gemini 2.0 Flash model.

## ✨ Features

- **Multi-format Support**: Handle PDF, DOC, DOCX, and image files (PNG, JPG, JPEG)
- **Smart Chunking**: Intelligent document segmentation for optimal context retrieval
- **Vector Database**: ChromaDB integration for efficient semantic search
- **Session Persistence**: SQLite database saves all your chats and documents
- **Session Isolation**: Each session has its own vector store - no data mixing
- **Vision Capabilities**: Ask questions about images using Gemini's vision model
- **Conversational AI**: Powered by Gemini 2.5 Flash for accurate and contextual responses
- **ChatGPT-like UI**: Clean, scrollable chat interface with sources inline
- **Document Viewer**: View your uploaded documents directly in the app
- **🎤 Voice Input**: Ask questions using your voice in any language (Speech-to-Text)
- **🌐 Multi-language Support**: Automatic language detection - supports English, Gujarati, Hindi, and 100+ languages

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12
- Google API Key (Gemini API)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_actual_api_key_here
     ```

   To get a Google API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key

## 📖 Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload documents**:
   - Enter your Google API key in the sidebar
   - Click "Choose a file" and select your document
   - Click "Process Document" to analyze it

3. **Chat with your documents**:
   - **Text Input**: Type your questions in the chat input box
   - **Voice Input**: Click the microphone icon to record your question in any language
   - The AI will automatically detect your language and respond in the same language
   - Responses are displayed as text in the chat interface with sources
   - For images, it uses vision capabilities to understand visual content
   - For text documents, it retrieves relevant chunks and generates contextual answers

4. **Supported Languages for Voice Input**:
   - English, Gujarati (ગુજરાતી), Hindi (हिंदी), Spanish, French, German, Arabic, Chinese, Japanese, Korean
   - Plus 100+ other languages via Google Speech Recognition

## 🏗️ Project Structure

```
smart_doc_agent/
├── app.py                          # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── document_processor.py      # Handle different document types
│   ├── chunking.py                # Smart document chunking
│   ├── vectordb.py                # ChromaDB vector store operations
│   ├── database.py                # SQLite session persistence
│   └── voice_handler.py           # Voice input/output (STT/TTS)
├── data/                          # SQLite database storage
├── chroma_db/                     # Vector database storage
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🔧 How It Works

### For Text Documents (PDF, DOC, DOCX):

1. **Document Processing**: Extracts text from uploaded files
2. **Smart Chunking**: Splits text into semantically meaningful chunks with overlap
3. **Embedding**: Converts chunks into vector embeddings using Google's embedding model
4. **Storage**: Stores embeddings in ChromaDB vector database
5. **Retrieval**: When you ask a question, finds the most relevant chunks
6. **Generation**: Uses Gemini 2.0 Flash to generate answers based on retrieved context

### For Images (PNG, JPG, JPEG):

1. **Image Processing**: Loads and stores the image
2. **Vision Analysis**: Uses Gemini's multimodal capabilities
3. **Direct Q&A**: Answers questions by analyzing the image content directly

### For Voice Input:

1. **Speech-to-Text (STT)**:
   - Records audio from browser microphone
   - Google Speech Recognition transcribes to text
   - Automatic language detection (supports 120+ languages)
   - Transcribed text is processed like any text query

2. **Language Detection**:
   - Detects language from user's question (voice or text)
   - Gemini AI responds in the detected language
   - Maintains language consistency throughout conversation

## 🛠️ Technologies Used

- **LangChain**: Framework for LLM applications
- **Google Gemini 2.5 Flash**: State-of-the-art language and vision model
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Web interface framework
- **PyPDF**: PDF processing
- **python-docx**: Word document processing
- **Pillow**: Image processing
- **SpeechRecognition**: Google Speech-to-Text for voice input
- **audio-recorder-streamlit**: Voice input component
- **langdetect**: Language detection library

## 📝 Notes

- The application maintains conversation history during a session
- You can upload multiple documents and they will all be indexed
- Use the "Clear All Documents" button to reset the session
- Source documents are shown for transparency in text-based queries

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.
