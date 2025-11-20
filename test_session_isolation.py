"""Test script to verify session isolation in vector store."""
import os
from dotenv import load_dotenv
from utils.vectordb import VectorStore
from langchain.docstore.document import Document

load_dotenv()

def test_session_isolation():
    """Test that different sessions have isolated vector stores."""
    print("Testing Session Isolation...")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_API_KEY in .env file")
        return

    # Create two sessions
    session1_id = "test_session_1"
    session2_id = "test_session_2"

    # Create vector stores for each session
    print(f"\n1. Creating vector store for session 1: {session1_id}")
    vs1 = VectorStore(api_key=api_key, session_id=session1_id)

    print(f"2. Creating vector store for session 2: {session2_id}")
    vs2 = VectorStore(api_key=api_key, session_id=session2_id)

    # Add documents to session 1
    print("\n3. Adding documents about Python to session 1...")
    docs1 = [
        Document(page_content="Python is a high-level programming language.", metadata={"source": "python_doc.pdf"}),
        Document(page_content="Python was created by Guido van Rossum.", metadata={"source": "python_doc.pdf"}),
    ]
    vs1.create_vectorstore(docs1)
    print("   ✓ Added Python documents to session 1")

    # Add documents to session 2
    print("\n4. Adding documents about JavaScript to session 2...")
    docs2 = [
        Document(page_content="JavaScript is a scripting language for web development.", metadata={"source": "js_doc.pdf"}),
        Document(page_content="JavaScript was created by Brendan Eich.", metadata={"source": "js_doc.pdf"}),
    ]
    vs2.create_vectorstore(docs2)
    print("   ✓ Added JavaScript documents to session 2")

    # Search in session 1 (should only find Python docs)
    print("\n5. Searching 'programming language creator' in session 1...")
    results1 = vs1.search_similar("programming language creator", k=2)
    print(f"   Found {len(results1)} results:")
    for i, doc in enumerate(results1):
        print(f"   - Result {i+1}: {doc.page_content[:50]}...")
        print(f"     Source: {doc.metadata.get('source')}")

    # Verify session 1 only has Python content
    has_python = any("Python" in doc.page_content for doc in results1)
    has_javascript = any("JavaScript" in doc.page_content for doc in results1)

    if has_python and not has_javascript:
        print("   ✅ Session 1 correctly isolated (only Python content)")
    else:
        print("   ❌ Session 1 NOT isolated (found JavaScript content)")
        return False

    # Search in session 2 (should only find JavaScript docs)
    print("\n6. Searching 'programming language creator' in session 2...")
    results2 = vs2.search_similar("programming language creator", k=2)
    print(f"   Found {len(results2)} results:")
    for i, doc in enumerate(results2):
        print(f"   - Result {i+1}: {doc.page_content[:50]}...")
        print(f"     Source: {doc.metadata.get('source')}")

    # Verify session 2 only has JavaScript content
    has_python2 = any("Python" in doc.page_content for doc in results2)
    has_javascript2 = any("JavaScript" in doc.page_content for doc in results2)

    if has_javascript2 and not has_python2:
        print("   ✅ Session 2 correctly isolated (only JavaScript content)")
    else:
        print("   ❌ Session 2 NOT isolated (found Python content)")
        return False

    # Clean up
    print("\n7. Cleaning up test sessions...")
    vs1.clear_vectorstore()
    vs2.clear_vectorstore()
    print("   ✓ Cleaned up")

    print("\n✅ All tests passed! Sessions are properly isolated.")
    return True

if __name__ == "__main__":
    test_session_isolation()
