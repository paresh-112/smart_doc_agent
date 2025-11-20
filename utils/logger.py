"""Logging configuration for Smart Document Agent."""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


class AppLogger:
    """Centralized logging system for the application."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logging system."""
        if not AppLogger._initialized:
            self._setup_logging()
            AppLogger._initialized = True

    def _setup_logging(self):
        """Configure logging with file and console handlers."""
        # Detect Streamlit Cloud environment
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src')

        # Create logs directory if it doesn't exist
        log_dir = "/tmp/logs" if is_streamlit_cloud else "./logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename with date
        log_filename = os.path.join(
            log_dir,
            f"app_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Get root logger
        self.logger = logging.getLogger('SmartDocAgent')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler with rotation (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("=" * 80)
        self.logger.info("Smart Document Agent - Logging System Initialized")
        self.logger.info("=" * 80)

    def get_logger(self, name: str = None):
        """Get logger instance for a specific module.

        Args:
            name: Module name (e.g., 'database', 'vectordb')

        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f'SmartDocAgent.{name}')
        return self.logger


# Create global logger instance
app_logger = AppLogger()


def get_logger(name: str = None):
    """Convenience function to get logger.

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return app_logger.get_logger(name)
