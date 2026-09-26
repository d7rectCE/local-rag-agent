"""Э10: the scanner of secrets and personal paths that CI runs over the repository."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_repo", Path(__file__).resolve().parents[1] / "scripts" / "check_repo.py")
check_repo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_repo)

# fake values are assembled at run time, so this file itself stays clean for the scanner
GH = "gh" + "p_" + "A1b2" * 9
HF = "hf" + "_" + "x9Y8" * 8
USER = "iv" + "anov"
BS = "\\"


def kinds(text: str) -> list[str]:
    return [k for _, k, _ in check_repo.scan_text(text)]


def test_tokens_and_keys_are_found_and_masked():
    found = check_repo.scan_text(f"token = '{GH}'\nHF_TOKEN={HF}\n" + "-----BEGIN " + "OPENSSH PRIVATE KEY-----")
    assert [k for _, k, _ in found] == ["GitHub token", "assigned secret", "Hugging Face token", "private key"]
    assert all(GH not in shown and HF not in shown for _, _, shown in found)  # never printed in full


def test_personal_paths():
    assert kinds("C:" + BS + "Users" + BS + USER + BS + "Desktop") == ["personal path"]
    assert kinds('"C:' + BS * 2 + "Users" + BS * 2 + USER + BS * 2 + 'x"') == ["personal path"]  # escaped in JSON
    assert kinds("/c/Users/" + USER + "/proj") == ["personal path"]
    assert kinds("/home/" + USER + "/.cache") == ["personal path"]
    generic = f"/home/runner/work /home/sandbox C:{BS}Users{BS}Public C:{BS}Users{BS}<name>{BS}x"
    assert kinds(generic) == []


def test_generic_values_and_the_marker_are_allowed():
    assert kinds("PASSWORD = 'x'\npassword: '<your password>'\napi_key = 'changeme-please-now'") == []
    assert kinds(f"token = '{GH}'  # noqa: secret") == []
