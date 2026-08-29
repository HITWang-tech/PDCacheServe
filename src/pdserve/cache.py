from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple

from .models import CacheTier, ModelLayout


@dataclass(frozen=True)
class CacheEntry:
    prefix_hash: str
    layout: ModelLayout
    token_count: int
    size_bytes: int
    tier: CacheTier
    location: str
    created_at: float
    last_access: float
    expires_at: float
    lease_count: int = 0

    @property
    def key(self) -> Tuple[str, str, str]:
        return self.prefix_hash, self.location, self.tier.value


class KVDirectory:
    """Central metadata directory; KV payloads remain distributed across tiers."""

    def __init__(
        self,
        capacities: Optional[Dict[CacheTier, int]] = None,
        clock=time.time,
    ) -> None:
        self.capacities = capacities or {
            CacheTier.GPU: 16 * 1024**3,
            CacheTier.CPU: 128 * 1024**3,
            CacheTier.SSD: 1024 * 1024**3,
        }
        self._clock = clock
        self._entries: Dict[Tuple[str, str, str], CacheEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        prefix_hash: str,
        layout: ModelLayout,
        token_count: int,
        tier: CacheTier,
        location: str,
        ttl_seconds: float = 300.0,
    ) -> CacheEntry:
        if not prefix_hash or token_count <= 0:
            raise ValueError("prefix_hash and positive token_count are required")
        now = self._clock()
        entry = CacheEntry(
            prefix_hash=prefix_hash,
            layout=layout,
            token_count=token_count,
            size_bytes=token_count * layout.kv_bytes_per_token,
            tier=tier,
            location=location,
            created_at=now,
            last_access=now,
            expires_at=now + ttl_seconds,
        )
        with self._lock:
            self._entries[entry.key] = entry
            self._evict_to_capacity(tier)
            return self._entries.get(entry.key, entry)

    def lookup(self, prefix_hash: str, layout: ModelLayout) -> List[CacheEntry]:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            matches = [
                item
                for item in self._entries.values()
                if item.prefix_hash == prefix_hash and item.layout.compatible_with(layout)
            ]
            matches.sort(
                key=lambda item: (
                    -item.token_count,
                    self._tier_rank(item.tier),
                    -item.last_access,
                )
            )
            return matches

    def acquire(self, key: Tuple[str, str, str]) -> CacheEntry:
        with self._lock:
            entry = self._entries[key]
            updated = replace(entry, lease_count=entry.lease_count + 1, last_access=self._clock())
            self._entries[key] = updated
            return updated

    def release(self, key: Tuple[str, str, str]) -> CacheEntry:
        with self._lock:
            entry = self._entries[key]
            updated = replace(entry, lease_count=max(0, entry.lease_count - 1))
            self._entries[key] = updated
            return updated

    def remove_location(self, location: str) -> int:
        with self._lock:
            keys = [key for key, value in self._entries.items() if value.location == location]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def entries(self) -> List[CacheEntry]:
        with self._lock:
            self._evict_expired(self._clock())
            return list(self._entries.values())

    def usage(self) -> Dict[str, int]:
        with self._lock:
            return {
                tier.value: sum(
                    item.size_bytes for item in self._entries.values() if item.tier == tier
                )
                for tier in CacheTier
            }

    def _evict_expired(self, now: float) -> None:
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now and entry.lease_count == 0:
                del self._entries[key]

    def _evict_to_capacity(self, tier: CacheTier) -> None:
        capacity = self.capacities.get(tier, 0)
        candidates: Iterable[CacheEntry] = sorted(
            (
                item
                for item in self._entries.values()
                if item.tier == tier and item.lease_count == 0
            ),
            key=lambda item: (item.last_access, item.created_at),
        )
        used = sum(item.size_bytes for item in self._entries.values() if item.tier == tier)
        for item in candidates:
            if used <= capacity:
                break
            used -= item.size_bytes
            self._entries.pop(item.key, None)

    @staticmethod
    def _tier_rank(tier: CacheTier) -> int:
        return {CacheTier.GPU: 0, CacheTier.CPU: 1, CacheTier.SSD: 2}[tier]
