from pdserve.cache import KVDirectory
from pdserve.models import CacheTier, ModelLayout


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_cache_lookup_requires_exact_model_layout():
    directory = KVDirectory()
    layout = ModelLayout("qwen", dtype="float16", block_size=16)
    directory.register("prefix", layout, 100, CacheTier.GPU, "d0")
    assert len(directory.lookup("prefix", layout)) == 1
    assert directory.lookup("prefix", ModelLayout("qwen", dtype="bfloat16")) == []
    assert directory.lookup("prefix", ModelLayout("other")) == []


def test_expired_entries_are_removed_unless_leased():
    clock = Clock()
    directory = KVDirectory(clock=clock)
    layout = ModelLayout("qwen")
    entry = directory.register("prefix", layout, 100, CacheTier.CPU, "host-0", ttl_seconds=5)
    directory.acquire(entry.key)
    clock.now = 110
    assert len(directory.lookup("prefix", layout)) == 1
    directory.release(entry.key)
    assert directory.lookup("prefix", layout) == []


def test_capacity_evicts_least_recently_used_entry():
    layout = ModelLayout("qwen", kv_bytes_per_token=10)
    directory = KVDirectory(
        capacities={CacheTier.GPU: 100, CacheTier.CPU: 1000, CacheTier.SSD: 1000}
    )
    directory.register("old", layout, 6, CacheTier.GPU, "d0")
    directory.register("new", layout, 6, CacheTier.GPU, "d1")
    assert directory.lookup("old", layout) == []
    assert len(directory.lookup("new", layout)) == 1


def test_remove_location_invalidates_worker_cache():
    layout = ModelLayout("qwen")
    directory = KVDirectory()
    directory.register("one", layout, 10, CacheTier.GPU, "d0")
    directory.register("two", layout, 10, CacheTier.GPU, "d1")
    assert directory.remove_location("d0") == 1
    assert len(directory.entries()) == 1
