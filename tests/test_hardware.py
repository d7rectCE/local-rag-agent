"""The GPU badge: detected on the machine the server runs on, never configured."""

from rag_agent import hardware


def test_the_card_that_computes_is_picked():
    assert hardware.pick(["AMD Radeon(TM) Graphics", "AMD Radeon RX 7900 XTX"]) == "AMD Radeon RX 7900 XTX"
    assert hardware.pick(["Microsoft Basic Display Adapter", "NVIDIA GeForce RTX 4090"]) == "NVIDIA GeForce RTX 4090"
    assert hardware.pick(["Intel(R) UHD Graphics 770"]) == "Intel(R) UHD Graphics 770"  # nothing better: still a GPU
    assert hardware.pick(["", "  "]) is None
    assert hardware.short_name("AMD Radeon RX 7900 XTX") == "RX 7900 XTX"
    assert hardware.short_name("NVIDIA GeForce RTX 4090") == "RTX 4090"


def test_windows_adapters_come_from_wmi(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_run", lambda args: "AMD Radeon(TM) Graphics\r\nAMD Radeon RX 7900 XTX\r\n")
    assert hardware._from_os() == "AMD Radeon RX 7900 XTX"


def test_gpu_name_is_detected(monkeypatch):
    monkeypatch.setattr(hardware, "_cached", None)
    monkeypatch.setattr(hardware, "_from_torch", lambda: None)
    monkeypatch.setattr(hardware, "_from_os", lambda: "NVIDIA GeForce RTX 3060")
    assert hardware.gpu_name() == "RTX 3060"
    monkeypatch.setattr(hardware, "_cached", None)
    monkeypatch.setattr(hardware, "_from_os", lambda: None)
    assert hardware.gpu_name() is None  # no card: no badge, and the next call asks again
