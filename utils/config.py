"""Configuration and environment validation for Smart Document Agent."""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger('config')


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class Config:
    """Application configuration manager."""

    # Required directories
    REQUIRED_DIRS = [
        './data',
        './chroma_db',
        './logs',
    ]

    # Required environment variables
    REQUIRED_ENV_VARS = [
        'GOOGLE_API_KEY',
    ]

    # Optional environment variables with defaults
    OPTIONAL_ENV_VARS = {
        'MAX_FILE_SIZE_MB': '10',
        'CHUNK_SIZE': '1000',
        'CHUNK_OVERLAP': '200',
        'MAX_SEARCH_RESULTS': '6',
        'DATABASE_PATH': './data/sessions.db',
        'CHROMA_DB_PATH': './chroma_db',
    }

    def __init__(self):
        """Initialize configuration."""
        logger.info("Initializing configuration...")
        load_dotenv()

        # Detect Streamlit Cloud environment
        self.is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src')
        if self.is_streamlit_cloud:
            logger.info("🌥️  Streamlit Cloud detected - using temporary storage")

        self._validate_and_setup()

    def _validate_and_setup(self):
        """Validate environment and setup required directories."""
        try:
            # Create required directories
            self._create_directories()

            # Validate environment variables
            self._validate_environment()

            # Load configuration
            self._load_config()

            logger.info("✅ Configuration validated successfully")

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {str(e)}")
            raise ConfigError(f"Configuration error: {str(e)}")

    def _create_directories(self):
        """Create required directories if they don't exist."""
        logger.info("Checking required directories...")

        if self.is_streamlit_cloud:
            # On Streamlit Cloud, use /tmp/ for writable storage
            logger.info("Using /tmp/ directory for Streamlit Cloud")
            tmp_dirs = ['/tmp/data', '/tmp/logs']
            for dir_path in tmp_dirs:
                path = Path(dir_path)
                if not path.exists():
                    logger.info(f"  Creating temp directory: {dir_path}")
                    path.mkdir(parents=True, exist_ok=True)
        else:
            # Local deployment - create normal directories
            for dir_path in self.REQUIRED_DIRS:
                path = Path(dir_path)
                if not path.exists():
                    logger.info(f"  Creating directory: {dir_path}")
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.debug(f"  Directory exists: {dir_path}")

        logger.info("✅ All required directories ready")

    def _validate_environment(self):
        """Validate required environment variables."""
        logger.info("Validating environment variables...")

        missing_vars = []
        for var in self.REQUIRED_ENV_VARS:
            value = os.getenv(var)
            if not value or value.strip() == '':
                missing_vars.append(var)
                logger.error(f"  ❌ Missing: {var}")
            else:
                # Don't log the actual value for security
                logger.info(f"  ✅ Found: {var}")

        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)

        logger.info("✅ All required environment variables present")

    def _load_config(self):
        """Load configuration values."""
        # Required values
        self.google_api_key = os.getenv('GOOGLE_API_KEY')

        # Optional values with defaults
        self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', self.OPTIONAL_ENV_VARS['MAX_FILE_SIZE_MB']))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', self.OPTIONAL_ENV_VARS['CHUNK_SIZE']))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', self.OPTIONAL_ENV_VARS['CHUNK_OVERLAP']))
        self.max_search_results = int(os.getenv('MAX_SEARCH_RESULTS', self.OPTIONAL_ENV_VARS['MAX_SEARCH_RESULTS']))

        # Use /tmp/ paths for Streamlit Cloud (writable), normal paths for local
        if self.is_streamlit_cloud:
            self.database_path = '/tmp/data/sessions.db'
            self.chroma_db_path = None  # In-memory for ChromaDB
        else:
            self.database_path = os.getenv('DATABASE_PATH', self.OPTIONAL_ENV_VARS['DATABASE_PATH'])
            self.chroma_db_path = os.getenv('CHROMA_DB_PATH', self.OPTIONAL_ENV_VARS['CHROMA_DB_PATH'])

        logger.info("Configuration loaded:")
        logger.info(f"  Deployment: {'Streamlit Cloud' if self.is_streamlit_cloud else 'Local'}")
        logger.info(f"  Chunk size: {self.chunk_size}")
        logger.info(f"  Chunk overlap: {self.chunk_overlap}")
        logger.info(f"  Max search results: {self.max_search_results}")
        logger.info(f"  Max file size: {self.max_file_size_mb}MB")
        logger.info(f"  Database path: {self.database_path}")
        logger.info(f"  ChromaDB path: {self.chroma_db_path if self.chroma_db_path else 'In-memory'}")

    def get_config_summary(self) -> Dict[str, str]:
        """Get configuration summary for display.

        Returns:
            Dictionary with configuration values
        """
        return {
            'chunk_size': str(self.chunk_size),
            'chunk_overlap': str(self.chunk_overlap),
            'max_search_results': str(self.max_search_results),
            'max_file_size_mb': str(self.max_file_size_mb),
            'database_path': self.database_path,
            'chroma_db_path': self.chroma_db_path,
        }

    @staticmethod
    def validate_file_size(file_size_bytes: int, max_size_mb: int = None) -> bool:
        """Validate uploaded file size.

        Args:
            file_size_bytes: File size in bytes
            max_size_mb: Maximum allowed size in MB (uses config default if None)

        Returns:
            True if valid, False otherwise
        """
        if max_size_mb is None:
            max_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', '10'))

        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size_bytes <= max_size_bytes

    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Ensure a directory exists, create if necessary.

        Args:
            directory: Directory path

        Returns:
            True if directory exists or was created successfully
        """
        try:
            path = Path(directory)
            if not path.exists():
                logger.info(f"Creating directory: {directory}")
                path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            return False


# Global config instance
_config_instance = None


def get_config() -> Config:
    """Get or create global config instance.

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
