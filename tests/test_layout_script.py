"""H8 layout detector script: subsets of a prepared split mix document categories."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "train_layout_detector", Path(__file__).resolve().parents[1] / "scripts" / "train_layout_detector.py")
tld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tld)


def test_limited_split_is_a_random_mix_not_a_prefix(tmp_path: Path):
    cats = ["financial_reports"] * 600 + ["manuals"] * 300 + ["patents"] * 100  # grouped as on disk
    images = [{"file": f"{k:06d}.jpg", "boxes": [], "labels": [], "doc_category": c, "doc": f"d{k}"}
              for k, c in enumerate(cats)]
    (tmp_path / "annotations.json").write_text(json.dumps({"classes": tld.CLASSES, "size": 640, "images": images}))
    sub = tld.LayoutDataset(tmp_path, limit=200)
    got = {c: sum(r["doc_category"] == c for r in sub.items) for c in ("financial_reports", "manuals", "patents")}
    assert len(sub) == 200 and all(v > 0 for v in got.values())
    assert [r["file"] for r in tld.LayoutDataset(tmp_path, limit=200).items] == [r["file"] for r in sub.items]
    assert len(tld.LayoutDataset(tmp_path)) == 1000
