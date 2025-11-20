"""Smart chunking system for documents."""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class SmartChunker:
    """Smart chunking for documents with semantic awareness."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """Initialize the smart chunker.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries to split on semantic boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
            ]
        )

    def chunk_pdf(self, pdf_data: dict) -> List[Document]:
        """Chunk PDF content smartly.

        Args:
            pdf_data: Dictionary containing PDF pages and content

        Returns:
            List of Document objects with chunks
        """
        documents = []

        # Combine all pages with page markers to maintain context across pages
        full_text_with_pages = []
        page_boundaries = []  # Track character positions where pages start
        current_position = 0

        for page_data in pdf_data['pages']:
            content = page_data['content']
            if content and content.strip():
                # Add page marker for reference
                page_text = f"\n[Page {page_data['page']}]\n{content}\n"
                full_text_with_pages.append(page_text)
                page_boundaries.append({
                    'page': page_data['page'],
                    'start': current_position,
                    'end': current_position + len(page_text)
                })
                current_position += len(page_text)

        # Combine all text
        full_text = "".join(full_text_with_pages)

        if not full_text.strip():
            return documents

        # Split the entire document into chunks with overlap
        # This allows chunks to span across pages naturally
        chunks = self.text_splitter.split_text(full_text)

        for chunk_idx, chunk in enumerate(chunks):
            # Only add non-empty chunks
            if chunk and chunk.strip():
                # Determine which page(s) this chunk belongs to
                # by checking if chunk contains page markers
                pages_in_chunk = []
                for page_info in page_boundaries:
                    if f"[Page {page_info['page']}]" in chunk:
                        pages_in_chunk.append(page_info['page'])

                # Use the first page or "multiple" if spans multiple pages
                if len(pages_in_chunk) == 1:
                    page_ref = pages_in_chunk[0]
                elif len(pages_in_chunk) > 1:
                    page_ref = f"{pages_in_chunk[0]}-{pages_in_chunk[-1]}"
                else:
                    page_ref = "unknown"

                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        'source': pdf_data['filename'],
                        'page': page_ref,
                        'chunk': chunk_idx,
                        'type': 'pdf'
                    }
                )
                documents.append(doc)

        return documents

    def chunk_docx(self, docx_data: dict) -> List[Document]:
        """Chunk DOCX content smartly.

        Args:
            docx_data: Dictionary containing DOCX paragraphs and content

        Returns:
            List of Document objects with chunks
        """
        documents = []

        # Combine all paragraphs into a single text
        full_text = "\n\n".join([p['content'] for p in docx_data['paragraphs'] if p['content'].strip()])

        # Skip if no content
        if not full_text or not full_text.strip():
            return documents

        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)

        for chunk_idx, chunk in enumerate(chunks):
            # Only add non-empty chunks
            if chunk and chunk.strip():
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        'source': docx_data['filename'],
                        'chunk': chunk_idx,
                        'type': 'docx'
                    }
                )
                documents.append(doc)

        return documents

    def chunk_document(self, doc_data: dict) -> List[Document]:
        """Chunk any document based on its type.

        Args:
            doc_data: Dictionary containing document data

        Returns:
            List of Document objects with chunks
        """
        doc_type = doc_data['type']

        if doc_type == 'pdf':
            return self.chunk_pdf(doc_data)
        elif doc_type == 'docx':
            return self.chunk_docx(doc_data)
        elif doc_type == 'image':
            # For images, we don't chunk - we'll use vision capabilities
            return [Document(
                page_content="",  # Empty content as we'll use the image directly
                metadata={
                    'source': doc_data['filename'],
                    'type': 'image',
                    'temp_path': doc_data['temp_path']
                }
            )]
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
