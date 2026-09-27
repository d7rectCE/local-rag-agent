"""Where the pixels of an image node live (ТЗ S4): the node records its source at parse time, and the
image is read back only when it is described or embedded, so the catalog never stores image bytes."""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from rag_agent.schema import Node

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MIN_SIDE = 48  # icons and bullets are not worth describing


def image_bytes(root: Path, node: Node) -> bytes | None:
    """PNG / JPEG bytes of the image a node stands for, or None when it cannot be read."""
    src = node.metadata.get("image_source")
    path = root / node.file_path
    try:
        if src == "file":
            return path.read_bytes()
        if src == "ipynb":
            return _notebook_image(path, node.location.cell, int(node.metadata.get("output", 0)))
        if src == "docx" or node.metadata.get("image_rel"):
            return _docx_image(path, node.metadata["image_rel"])
        if src == "pdf" or (node.file_type == "pdf" and node.metadata.get("bbox")):
            return pdf_crop(path, node.location.page, node.metadata.get("bbox"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None
    return None


def _notebook_image(path: Path, cell: int | None, output: int) -> bytes | None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    if cell is None or not 1 <= cell <= len(cells):
        return None
    k = 0
    for out in cells[cell - 1].get("outputs", []):
        data = out.get("data") or {}
        mime = next((m for m in ("image/png", "image/jpeg") if m in data), None)
        if mime is None:
            continue
        if k == output:
            b64 = data[mime]
            return base64.b64decode("".join(b64) if isinstance(b64, list) else b64)
        k += 1
    return None


def notebook_images(outputs: list[dict]) -> int:
    return sum(1 for out in outputs if any(m in (out.get("data") or {}) for m in ("image/png", "image/jpeg")))


def _docx_image(path: Path, rel_id: str) -> bytes | None:
    with zipfile.ZipFile(path) as zf:
        rels = zf.read("word/_rels/document.xml.rels").decode("utf-8")
        m = re.search(rf'Id="{re.escape(rel_id)}"[^>]*Target="([^"]+)"', rels) or \
            re.search(rf'Target="([^"]+)"[^>]*Id="{re.escape(rel_id)}"', rels)
        if not m:
            return None
        target = str(PurePosixPath("word") / m.group(1)) if not m.group(1).startswith("/") else m.group(1).lstrip("/")
        target = str(PurePosixPath(target)).replace("word/../", "")
        data = zf.read(target)
    return data if data[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1") or data[:3] == b"\xff\xd8\xff" else None


def pdf_crop(path: Path, page: int | None, bbox: list[float] | None, scale: float = 2.0) -> bytes | None:
    """A PDF picture: the page rendered and cut to the picture's box (Docling's box: left, top, right,
    bottom in points, origin at the bottom left when top > bottom). Without a box: the whole page."""
    import pypdfium2 as pdfium

    if not page:
        return None
    pdf = pdfium.PdfDocument(str(path))
    try:
        p = pdf[page - 1]
        width, height = p.get_size()
        img = p.render(scale=scale).to_pil()
    finally:
        pdf.close()
    if bbox:
        l, t, r, b = bbox
        top, bottom = (height - t, height - b) if t > b else (t, b)
        box = [int(max(0, v * scale)) for v in (l, top, r, bottom)]
        if box[2] - box[0] >= MIN_SIDE and box[3] - box[1] >= MIN_SIDE:
            img = img.crop(box)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def pdf_page(path: Path, page: int, scale: float = 1.5) -> bytes | None:
    return pdf_crop(path, page, None, scale)


def big_enough(data: bytes) -> bool:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            return min(im.size) >= MIN_SIDE
    except OSError:
        return False
