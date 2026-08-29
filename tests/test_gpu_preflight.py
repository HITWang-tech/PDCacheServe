import sys
from types import SimpleNamespace

from pdserve.gpu_preflight import inspect_devices


class FakeCuda:
    def __init__(self, peer_access: bool = True):
        self.peer_access = peer_access

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def get_device_name(self, device):
        return f"GPU-{device}"

    def get_device_properties(self, device):
        return SimpleNamespace(total_memory=24 * 1024**3)

    def can_device_access_peer(self, source, target):
        return self.peer_access


def test_preflight_accepts_bidirectional_peer_access(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda(True)))
    result = inspect_devices([0, 1])
    assert result["eligible"] is True
    assert result["peer_access"] == {"0->1": True, "1->0": True}


def test_preflight_rejects_no_peer_access(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda(False)))
    result = inspect_devices([0, 1])
    assert result["eligible"] is False
    assert "unavailable" in result["reason"]
