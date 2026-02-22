"""
ingestion/parsers/base.py
─────────────────────────
Abstract base class for all format plugins.
To add a new format: implement this interface, register with PluginRegistry.
No changes to any other part of the pipeline required.
"""
from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

from src.core.logging import get_logger
from src.core.models import Chunk, DocumentFormat, ParsedDocument

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk Config
# ─────────────────────────────────────────────────────────────────────────────

class ChunkConfig:
    def __init__(
        self,
        child_chunk_size: int = 400,
        parent_chunk_size: int = 1500,
        chunk_overlap: int = 50,
    ):
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.chunk_overlap = chunk_overlap


# ─────────────────────────────────────────────────────────────────────────────
# Base Plugin
# ─────────────────────────────────────────────────────────────────────────────

class BaseFormatPlugin(ABC):
    """
    Implement this class to add support for a new file format.

    Steps:
      1. Create a new file in src/ingestion/parsers/
      2. Implement parse() and chunk()
      3. Plugin is auto-discovered — no manual registration needed

    Example:
        class DocxPlugin(BaseFormatPlugin):
            supported_extensions = ['.docx', '.doc']
            ...
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """File extensions this plugin handles. E.g., ['.pdf']"""

    @property
    def document_format(self) -> DocumentFormat:
        """Override to return the DocumentFormat enum value."""
        return DocumentFormat.UNKNOWN

    @property
    def default_namespace_hint(self) -> Optional[str]:
        """
        Optional hint for namespace routing.
        E.g., a legal contract plugin might hint 'LEGAL'.
        """
        return None

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """
        Parse raw file bytes into a normalized ParsedDocument.
        - text_blocks → will be chunked and embedded
        - tables       → will be stored in PostgreSQL/DuckDB
        - images       → will be stored in MinIO (CLIP embedding optional)
        """

    @abstractmethod
    def chunk(
        self,
        doc: ParsedDocument,
        config: ChunkConfig,
    ) -> List[Chunk]:
        """
        Apply format-specific chunking strategy.
        Returns list of Chunk objects ready for enrichment and embedding.
        Parent-child chunking should be implemented here.
        """

    def enrich_metadata(self, chunk: Chunk) -> Chunk:
        """
        Optional format-specific metadata additions.
        Override to add format-specific fields beyond the standard schema.
        Default: no-op.
        """
        return chunk

    def can_handle(self, filename: str) -> bool:
        """Returns True if this plugin can handle the given filename."""
        suffix = Path(filename).suffix.lower()
        return suffix in [ext.lower() for ext in self.supported_extensions]


# ─────────────────────────────────────────────────────────────────────────────
# Plugin Registry — Auto-Discovery
# ─────────────────────────────────────────────────────────────────────────────

class PluginRegistry:
    """
    Central registry for all format plugins.
    Plugins are auto-discovered from the parsers package.
    Manual registration also supported via register().
    """
    _plugins: Dict[str, BaseFormatPlugin] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, plugin: BaseFormatPlugin) -> None:
        """Manually register a plugin."""
        for ext in plugin.supported_extensions:
            cls._plugins[ext.lower()] = plugin
            logger.info("Plugin registered", extension=ext, plugin=type(plugin).__name__)

    @classmethod
    def get(cls, filename: str) -> BaseFormatPlugin:
        """
        Get the correct plugin for a filename.
        Falls back to auto-discovery if not yet initialized.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename must be a non-empty string")
        if not cls._discovered:
            cls._auto_discover()

        suffix = Path(filename).suffix.lower()
        plugin = cls._plugins.get(suffix)

        if plugin is None:
            raise ValueError(
                f"No plugin registered for extension '{suffix}'. "
                f"Registered extensions: {list(cls._plugins.keys())}"
            )
        return plugin

    @classmethod
    def _auto_discover(cls) -> None:
        """
        Automatically import all modules in the parsers package.
        Any BaseFormatPlugin subclass is registered automatically.
        """
        import src.ingestion.parsers as parsers_pkg
        pkg_path = Path(parsers_pkg.__file__).parent

        for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            if module_name == "base":
                continue
            try:
                module = importlib.import_module(f"src.ingestion.parsers.{module_name}")
                # Find all BaseFormatPlugin subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseFormatPlugin)
                        and attr is not BaseFormatPlugin
                    ):
                        cls.register(attr())
            except Exception as e:
                logger.warning("Failed to load plugin module", module=module_name, error=str(e))

        cls._discovered = True

    @classmethod
    def list_supported_extensions(cls) -> List[str]:
        if not cls._discovered:
            cls._auto_discover()
        return list(cls._plugins.keys())