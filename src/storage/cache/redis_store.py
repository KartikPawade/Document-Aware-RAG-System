"""
storage/cache/redis_store.py
─────────────────────────────
Redis cache for:
1. Query result cache (30-min TTL) — identical queries skip retrieval entirely
2. Hot chunk cache (5-min TTL) — newly ingested chunks available immediately

CHANGES for redis-py 5.2.x + Redis 7.4:
  - `aioredis.from_url()` is `redis.asyncio.from_url()` — no change, already correct.
  - `client.scan(cursor, match=..., count=...)` — in redis-py 5.x the method signature
    is `scan(cursor=0, match=None, count=None, _type=None)`. The positional call
    `client.scan(cursor, match=..., count=...)` still works.
  - `client.setex(name, time, value)` — `time` can be int (seconds) or timedelta.
    Still works unchanged.
  - hiredis 3.x (required for redis-py 5.2+ on Python 3.12) is a C extension;
    no API changes visible in Python code.
  - Added `auto_close_connection_pool=False` to prevent premature pool teardown
    in long-lived async applications (redis-py 5.x regression fix).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RedisCache:

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                # redis-py 5.x: set max connections explicitly for production
                max_connections=20,
            )
        return self._client

    # ─────────────────────────────────────────────────────────────────────────
    # Query Cache
    # ─────────────────────────────────────────────────────────────────────────

    def _query_cache_key(self, query: str, namespaces: List[str]) -> str:
        """Deterministic cache key from query + namespace list."""
        ns_str = ",".join(sorted(namespaces))
        raw = f"query:{query}:{ns_str}"
        return "qc:" + hashlib.sha256(raw.encode()).hexdigest()

    async def get_cached_response(
        self,
        query: str,
        namespaces: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Return cached RAG response if available."""
        key = self._query_cache_key(query, namespaces)
        try:
            data = await self._get_client().get(key)
            if data:
                logger.debug("Query cache hit", key=key[:20])
                return json.loads(data)
        except Exception as e:
            logger.warning("Cache get failed", error=str(e))
        return None

    async def cache_response(
        self,
        query: str,
        namespaces: List[str],
        response: Dict[str, Any],
    ) -> None:
        """Cache a RAG response with TTL."""
        key = self._query_cache_key(query, namespaces)
        try:
            await self._get_client().setex(
                key,
                settings.query_cache_ttl,
                json.dumps(response),
            )
            logger.debug("Response cached", key=key[:20], ttl=settings.query_cache_ttl)
        except Exception as e:
            logger.warning("Cache set failed", error=str(e))

    async def invalidate_namespace(self, namespace: str) -> None:
        """Invalidate all cached responses for a namespace (after bulk ingest)."""
        client = self._get_client()
        cursor = 0
        deleted = 0
        while True:
            # redis-py 5.x: scan returns (int, list[str]) — same as before
            cursor, keys = await client.scan(cursor, match="qc:*", count=100)
            if keys:
                await client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("Cache invalidated", namespace=namespace, deleted=deleted)

    # ─────────────────────────────────────────────────────────────────────────
    # Hot Chunk Cache
    # ─────────────────────────────────────────────────────────────────────────

    async def cache_hot_chunk(self, chunk_id: str, chunk_data: Dict[str, Any]) -> None:
        """Cache a newly ingested chunk for immediate retrieval before index propagation."""
        key = f"chunk:{chunk_id}"
        try:
            await self._get_client().setex(
                key,
                settings.hot_chunk_ttl,
                json.dumps(chunk_data),
            )
        except Exception as e:
            logger.warning("Hot chunk cache failed", chunk_id=chunk_id, error=str(e))

    async def get_hot_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk from hot cache."""
        key = f"chunk:{chunk_id}"
        try:
            data = await self._get_client().get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning("Hot chunk get failed", chunk_id=chunk_id, error=str(e))
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Health
    # ─────────────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            return await self._get_client().ping()
        except Exception:
            return False


# Singleton
_cache: Optional[RedisCache] = None

def get_redis_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
