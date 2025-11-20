"""Database module for session persistence using SQLite."""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
from utils.logger import get_logger

logger = get_logger('database')


class SessionDatabase:
    """Manage sessions, chat history, and documents using SQLite."""

    def __init__(self, db_path: str = "./data/sessions.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        logger.info(f"Initializing database at: {db_path}")

        try:
            # Create data directory if it doesn't exist
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                logger.info(f"Creating database directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)

            # Initialize database
            self._init_db()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}", exc_info=True)
            raise

    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                doc_data TEXT,
                chunks_count INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)

        conn.commit()
        conn.close()

    def create_session(self, session_id: str, title: Optional[str] = None) -> bool:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            title: Optional session title

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if title is None:
                title = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            cursor.execute("""
                INSERT INTO sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (session_id, title))

            conn.commit()
            conn.close()
            logger.info(f"Created session: {session_id[:8]}... - {title}")
            return True
        except Exception as e:
            logger.error(f"Error creating session {session_id[:8]}...: {str(e)}", exc_info=True)
            return False

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions ordered by update time.

        Returns:
            List of session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT s.session_id, s.title, s.created_at, s.updated_at, s.is_active,
                   COUNT(DISTINCT d.doc_id) as doc_count,
                   COUNT(DISTINCT m.msg_id) as msg_count
            FROM sessions s
            LEFT JOIN documents d ON s.session_id = d.session_id
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append({
                'session_id': row[0],
                'title': row[1],
                'created_at': row[2],
                'updated_at': row[3],
                'is_active': bool(row[4]),
                'doc_count': row[5],
                'msg_count': row[6]
            })

        return sessions

    def save_document(self, session_id: str, filename: str, file_type: str,
                     doc_data: Dict, chunks_count: int) -> bool:
        """Save document metadata to database.

        Args:
            session_id: Session identifier
            filename: Document filename
            file_type: Document type (pdf, docx, image)
            doc_data: Document data dictionary
            chunks_count: Number of chunks created

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clear previous documents for this session (single doc mode)
            cursor.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))

            cursor.execute("""
                INSERT INTO documents (session_id, filename, file_type, doc_data, chunks_count)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, filename, file_type, json.dumps(doc_data), chunks_count))

            # Update session timestamp
            cursor.execute("""
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (session_id,))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False

    def get_session_document(self, session_id: str) -> Optional[Dict]:
        """Get current document for a session.

        Args:
            session_id: Session identifier

        Returns:
            Document dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT filename, file_type, doc_data, chunks_count, uploaded_at
            FROM documents
            WHERE session_id = ?
            ORDER BY uploaded_at DESC
            LIMIT 1
        """, (session_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'filename': row[0],
                'file_type': row[1],
                'doc_data': json.loads(row[2]) if row[2] else None,
                'chunks_count': row[3],
                'uploaded_at': row[4]
            }
        return None

    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """Save a chat message.

        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO messages (session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, role, content))

            # Update session timestamp
            cursor.execute("""
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (session_id,))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving message: {e}")
            return False

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2]
            })

        return messages

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title.

        Args:
            session_id: Session identifier
            title: New title

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE sessions
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (title, session_id))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating session title: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete messages
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            # Delete documents
            cursor.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))

            # Delete session
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

            conn.commit()
            conn.close()
            logger.info(f"Deleted session: {session_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id[:8]}...: {str(e)}", exc_info=True)
            return False

    def clear_session_messages(self, session_id: str) -> bool:
        """Clear all messages in a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing messages: {e}")
            return False

    def clear_all_sessions(self) -> bool:
        """Delete all sessions and their data.

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete all messages
            cursor.execute("DELETE FROM messages")

            # Delete all documents
            cursor.execute("DELETE FROM documents")

            # Delete all sessions
            cursor.execute("DELETE FROM sessions")

            conn.commit()
            conn.close()
            logger.info("Cleared all sessions from database")
            return True
        except Exception as e:
            logger.error(f"Error clearing all sessions: {str(e)}", exc_info=True)
            return False
