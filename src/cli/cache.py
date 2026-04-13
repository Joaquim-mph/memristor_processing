"""
Caching layer for CLI commands.

Provides thread-safe caching of loaded data with TTL-based expiration,
file modification tracking, and LRU eviction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Callable
import polars as pl
from datetime import datetime, timedelta
import threading
import hashlib


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    file_mtime: Optional[float]  # File modification time for invalidation
    access_count: int
    size_bytes: int  # Approximate memory size


class CacheStats:
    """Cache statistics for monitoring"""
    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.invalidations: int = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return f"Hits: {self.hits}, Misses: {self.misses}, Hit Rate: {self.hit_rate():.1%}"


class DataCache:
    """Thread-safe cache for loaded data with TTL and invalidation"""

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_items: int = 50,
        max_size_mb: int = 500
    ):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached items
            max_items: Maximum number of items to cache
            max_size_mb: Maximum cache size in megabytes
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_items = max_items
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._stats = CacheStats()

    def _generate_key(self, file_path: Path, **kwargs) -> str:
        """Generate unique cache key from file path and parameters"""
        # Include file path and any filtering parameters
        key_parts = [str(file_path.resolve())]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired"""
        age = datetime.now() - entry.created_at
        return age > self._ttl

    def _is_file_modified(self, entry: CacheEntry, file_path: Path) -> bool:
        """Check if source file has been modified since caching"""
        if entry.file_mtime is None:
            return False
        if not file_path.exists():
            return True
        return file_path.stat().st_mtime > entry.file_mtime

    def _evict_lru(self):
        """Evict least recently used item"""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]
        self._stats.evictions += 1

    def _enforce_size_limit(self):
        """Evict items if cache exceeds size limit"""
        current_size = sum(entry.size_bytes for entry in self._cache.values())
        while current_size > self._max_size_bytes and self._cache:
            self._evict_lru()
            current_size = sum(entry.size_bytes for entry in self._cache.values())

    def get(
        self,
        file_path: Path,
        loader_fn: Callable,
        **kwargs
    ) -> Any:
        """
        Get data from cache or load using loader function.

        Args:
            file_path: Path to data file
            loader_fn: Function to load data if not cached
            **kwargs: Additional parameters for cache key generation

        Returns:
            Cached or loaded data
        """
        key = self._generate_key(file_path, **kwargs)

        with self._lock:
            # Check if in cache
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if self._is_expired(entry):
                    del self._cache[key]
                    self._stats.invalidations += 1
                # Check file modification
                elif self._is_file_modified(entry, file_path):
                    del self._cache[key]
                    self._stats.invalidations += 1
                else:
                    # Cache hit!
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self._stats.hits += 1
                    return entry.value

            # Cache miss - need to load
            self._stats.misses += 1

        # Load data (outside lock to avoid blocking)
        data = loader_fn(file_path, **kwargs)

        # Estimate size (rough approximation)
        if isinstance(data, pl.DataFrame):
            size_bytes = data.estimated_size()
        else:
            size_bytes = 1024  # Default estimate

        # Store in cache
        with self._lock:
            # Enforce item limit
            if len(self._cache) >= self._max_items:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                file_mtime=file_path.stat().st_mtime if file_path.exists() else None,
                access_count=1,
                size_bytes=size_bytes
            )
            self._cache[key] = entry

            # Enforce size limit
            self._enforce_size_limit()

        return data

    def invalidate(self, file_path: Optional[Path] = None):
        """
        Invalidate cache entries.

        Args:
            file_path: If provided, only invalidate entries for this file.
                      If None, clear entire cache.
        """
        with self._lock:
            if file_path is None:
                # Clear entire cache
                self._stats.invalidations += len(self._cache)
                self._cache.clear()
            else:
                # Invalidate entries matching file path
                resolved_path = str(file_path.resolve())
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if resolved_path in str(entry.key)
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                    self._stats.invalidations += 1

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats

    def get_info(self) -> dict:
        """Get detailed cache information"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            return {
                "item_count": len(self._cache),
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "utilization": total_size / self._max_size_bytes if self._max_size_bytes > 0 else 0,
                "stats": str(self._stats)
            }

    def cleanup_expired(self):
        """Remove expired entries from cache"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.invalidations += 1


# Global cache instance
_cache: Optional[DataCache] = None


def get_cache() -> DataCache:
    """
    Get or create global cache instance.

    Note: Cache is process-local and will be recreated in child processes
    after fork (e.g., in multiprocessing workers). This is intentional to
    avoid sharing threading locks and other fork-unsafe resources.
    """
    global _cache
    if _cache is None:
        from src.cli.main import get_config
        config = get_config()
        _cache = DataCache(
            ttl_seconds=config.cache_ttl,
            max_items=config.cache_max_items,
            max_size_mb=config.cache_max_size_mb
        )
    return _cache


def reset_cache():
    """
    Reset global cache instance.

    Useful for multiprocessing scenarios where child processes need
    to reinitialize the cache after fork.
    """
    global _cache
    _cache = None


def clear_cache():
    """Clear global cache"""
    global _cache
    if _cache is not None:
        _cache.invalidate()


# Helper functions for common use cases

def load_history_cached(
    history_file: Path,
    filters: Optional[dict] = None
) -> pl.DataFrame:
    """
    Load chip history with caching.

    Args:
        history_file: Path to history Parquet file
        filters: Optional filters to apply

    Returns:
        Loaded and filtered history DataFrame
    """
    cache = get_cache()

    def loader(file_path: Path, **kwargs) -> pl.DataFrame:
        # Load from disk
        history = pl.read_parquet(file_path)

        # Apply filters if provided
        filter_dict = kwargs.get('filters')
        if filter_dict:
            for key, value in filter_dict.items():
                if key in history.columns:
                    history = history.filter(pl.col(key) == value)

        return history

    return cache.get(
        history_file,
        loader,
        filters=filters or {}
    )


def load_parquet_cached(file_path: Path) -> pl.DataFrame:
    """Load Parquet file with caching"""
    cache = get_cache()
    return cache.get(
        file_path,
        lambda p, **kw: pl.read_parquet(p)
    )
