import os
import logging
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles downloading and caching of model files from URLs."""

    # Cache directory for downloaded models
    CACHE_DIR = Path("model_cache")

    @classmethod
    def _is_url(cls, path: str) -> bool:
        """Check if a path is a URL."""
        try:
            result = urlparse(path)
            return result.scheme in ("http", "https", "s3")
        except Exception:
            return False

    @classmethod
    def _ensure_cache_dir(cls):
        """Ensure the cache directory exists."""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_cache_path(cls, url: str) -> Path:
        """Get the local cache path for a URL."""
        # Extract filename from URL
        parsed = urlparse(url)
        filename = Path(parsed.path).name

        if not filename:
            raise ValueError(f"Could not extract filename from URL: {url}")

        return cls.CACHE_DIR / filename

    @classmethod
    def _download_file(cls, url: str, destination: Path) -> Path:
        """Download a file from URL to destination."""
        logger.info(f"Downloading model from {url}")
        logger.info(f"Saving to {destination}")

        try:
            urlretrieve(url, destination)
            logger.info(f"Successfully downloaded {destination.name}")
            return destination
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    @classmethod
    def get_local_path(cls, path_or_url: str) -> str:
        """
        Get a local file path for the given path or URL.

        If the input is a URL, downloads the file to cache and returns the local path.
        If the input is already a local path, returns it as-is.

        Args:
            path_or_url: Either a local file path or a URL to a model file

        Returns:
            Local file path as a string
        """
        # If it's not a URL, return as-is
        if not cls._is_url(path_or_url):
            return path_or_url

        # Ensure cache directory exists
        cls._ensure_cache_dir()

        # Get cache path
        cache_path = cls._get_cache_path(path_or_url)

        # If file already cached, return it
        if cache_path.exists():
            logger.info(f"Using cached model: {cache_path.name}")
            return str(cache_path)

        # Download the file
        cls._download_file(path_or_url, cache_path)

        return str(cache_path)
