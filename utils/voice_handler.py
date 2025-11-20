"""Voice processing utilities for speech-to-text."""
import os
import tempfile
import atexit
from pathlib import Path
import speech_recognition as sr
from langdetect import detect, LangDetectException
from typing import Optional, Tuple, List
from utils.logger import get_logger

logger = get_logger('voice')


class VoiceHandler:
    """Handle voice input with multi-language support."""

    # Language code mapping for common languages
    LANGUAGE_CODES = {
        'en': 'English',
        'gu': 'Gujarati',
        'hi': 'Hindi',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh-cn': 'Chinese',
        'ar': 'Arabic',
        'ja': 'Japanese',
        'ko': 'Korean',
    }

    def __init__(self):
        """Initialize voice handler."""
        logger.info("Initializing VoiceHandler")
        self.recognizer = sr.Recognizer()
        self.temp_files: List[str] = []  # Track temp files for cleanup

        # Register cleanup on exit
        atexit.register(self.cleanup_temp_files)
        logger.info("✅ VoiceHandler initialized")

    def transcribe_audio(self, audio_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        """Transcribe audio to text using Google Speech Recognition.

        Args:
            audio_bytes: Audio data in bytes

        Returns:
            Tuple of (transcribed_text, detected_language_code)
        """
        tmp_path = None
        try:
            logger.info(f"Transcribing audio ({len(audio_bytes)} bytes)")

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
                self.temp_files.append(tmp_path)  # Track for cleanup

            # Load audio file
            with sr.AudioFile(tmp_path) as source:
                audio_data = self.recognizer.record(source)

            # Try to recognize speech
            # Google Speech Recognition supports automatic language detection
            text = self.recognizer.recognize_google(audio_data)

            # Detect language from transcribed text
            try:
                lang_code = detect(text)
            except LangDetectException:
                lang_code = 'en'  # Default to English

            logger.info(f"✅ Transcribed ({lang_code}): {text[:50]}...")

            # Clean up temp file immediately
            self._cleanup_file(tmp_path)

            return text, lang_code

        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            if tmp_path:
                self._cleanup_file(tmp_path)
            return None, None
        except sr.RequestError as e:
            logger.error(f"Speech recognition API error: {str(e)}")
            if tmp_path:
                self._cleanup_file(tmp_path)
            return None, None
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            if tmp_path:
                self._cleanup_file(tmp_path)
            return None, None

    def detect_language_from_text(self, text: str) -> str:
        """Detect language from text.

        Args:
            text: Input text

        Returns:
            Language code (e.g., 'en', 'gu', 'hi')
        """
        try:
            lang_code = detect(text)
            return lang_code
        except LangDetectException:
            return 'en'  # Default to English

    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name from code.

        Args:
            lang_code: Language code (e.g., 'en', 'gu')

        Returns:
            Language name (e.g., 'English', 'Gujarati')
        """
        # Extended language names for common languages
        extended_names = {
            **self.LANGUAGE_CODES,
            'mr': 'Marathi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'pa': 'Punjabi',
            'ur': 'Urdu',
            'pt': 'Portuguese',
            'it': 'Italian',
            'ru': 'Russian',
        }
        return extended_names.get(lang_code, lang_code.upper())

    def _cleanup_file(self, file_path: str) -> None:
        """Clean up a single temp file.

        Args:
            file_path: Path to temp file
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
                # Remove from tracking list
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")

    def cleanup_temp_files(self) -> None:
        """Clean up all tracked temporary files."""
        if self.temp_files:
            logger.info(f"Cleaning up {len(self.temp_files)} temp files")
            for file_path in self.temp_files.copy():  # Use copy to avoid modification during iteration
                self._cleanup_file(file_path)
            logger.info("✅ Temp files cleaned up")
