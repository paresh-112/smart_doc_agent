"""Test script for database functionality."""
from utils.database import SessionDatabase
import uuid

def test_database():
    """Test database operations."""
    print("Testing Session Database...")

    # Initialize database
    db = SessionDatabase("./data/test.db")
    print("✓ Database initialized")

    # Create a test session
    session_id = str(uuid.uuid4())
    db.create_session(session_id, "Test Session")
    print(f"✓ Created session: {session_id}")

    # Save a test document
    doc_data = {
        'filename': 'test.pdf',
        'type': 'pdf',
        'content': 'Test content'
    }
    db.save_document(session_id, 'test.pdf', 'pdf', doc_data, 10)
    print("✓ Saved document")

    # Save test messages
    db.save_message(session_id, 'user', 'Hello, can you help me?')
    db.save_message(session_id, 'assistant', 'Of course! How can I help you today?')
    print("✓ Saved messages")

    # Retrieve session
    sessions = db.get_all_sessions()
    print(f"✓ Retrieved {len(sessions)} session(s)")

    # Retrieve messages
    messages = db.get_session_messages(session_id)
    print(f"✓ Retrieved {len(messages)} message(s)")

    # Retrieve document
    doc = db.get_session_document(session_id)
    if doc:
        print(f"✓ Retrieved document: {doc['filename']}")

    # Update session title
    db.update_session_title(session_id, "Updated Test Session")
    print("✓ Updated session title")

    # Clean up
    db.delete_session(session_id)
    print("✓ Deleted test session")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_database()
