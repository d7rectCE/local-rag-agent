"""H8: fine-tune a layout detector (RT-DETRv2) on DocLayNet and compare it with
Docling's layout model (ТЗ S1 "CV-компонент проекта").

    # 1. DocLayNet parquet (HF cache) -> 640x640 JPEG + boxes
    python scripts/train_layout_detector.py prepare --out C:/ml-cache/doclaynet640
    # 2. fine-tune a COCO-pretrained RT-DETRv2 (checkpoint every 500 steps)
    python scripts/train_layout_detector.py train --data C:/ml-cache/doclaynet640 --out runs/layout/rtdetr_v2_r50
    #    pause: save a checkpoint and exit; continue later from the same batch
    python scripts/train_layout_detector.py stop --out runs/layout/rtdetr_v2_r50
    python scripts/train_layout_detector.py train --data C:/ml-cache/doclaynet640 --out runs/layout/rtdetr_v2_r50 --resume
    # 3. mAP on the DocLayNet test split: ours and Docling's (heron)
    python scripts/train_layout_detector.py evaluate --data C:/ml-cache/doclaynet640 --model runs/layout/rtdetr_v2_r50/best
    python scripts/train_layout_detector.py evaluate --data C:/ml-cache/doclaynet640 --model <docling models dir>/<heron>

DocLayNet v1.2 is read from the local Hugging Face cache (docling-project/DocLayNet-v1.2).
Without validation shards, the last train shard is held out as validation.
All models are loaded from local files only.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture",
           "Section-header", "Table", "Text", "Title"]  # DocLayNet category_id 1..11
DATASET = "docling-project/DocLayNet-v1.2"
BASE_MODEL = "PekingU/rtdetr_v2_r50vd"


def norm_label(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


# --------------------------------------------------------------------------- prepare


def dataset_files() -> dict[str, list[Path]]:
    """Parquet shards present in the local HF cache (a partial download is fine)."""
    from huggingface_hub import scan_cache_dir

    repo = next((r for r in scan_cache_dir().repos if r.repo_id == DATASET and r.repo_type == "dataset"), None)
    if repo is None:
        raise SystemExit(f'{DATASET} is not in the HF cache: hf download {DATASET} --repo-type dataset --include "data/*"')
    files: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}
    for rev in repo.revisions:
        for f in rev.files:
            name = f.file_name
            for split in files:
                if name.startswith(f"{split}-") and name.endswith(".parquet"):
                    files[split].append(Path(f.file_path))
    files = {split: sorted(set(paths)) for split, paths in files.items()}
    if not files["validation"] and len(files["train"]) > 1:
        files["validation"] = [files["train"].pop()]  # hold out one train shard
    return files


def prepare(out: Path, size: int, limit: int | None) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    for split, paths in dataset_files().items():
        target = out / split
        target.mkdir(parents=True, exist_ok=True)
        records, n = [], 0
        for path in paths:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["image", "bboxes", "category_id", "metadata"])
                for row in table.to_pylist():
                    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                    sx, sy = size / img.width, size / img.height
                    name = f"{n:06d}.jpg"
                    img.resize((size, size), Image.BILINEAR).save(target / name, quality=90)
                    boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in row["bboxes"]]
                    keep = [i for i, b in enumerate(boxes) if b[2] > 1 and b[3] > 1]
                    records.append({
                        "file": name,
                        "boxes": [[round(v, 2) for v in boxes[i]] for i in keep],
                        "labels": [int(row["category_id"][i]) - 1 for i in keep],
                        "doc_category": row["metadata"].get("doc_category"),
                    })
                    n += 1
                    if limit and n >= limit:
                        break
                if limit and n >= limit:
                    break
            if limit and n >= limit:
                break
        (target / "annotations.json").write_text(json.dumps({"classes": CLASSES, "size": size, "images": records}))
        print(f"{split}: {n} pages -> {target}")


# --------------------------------------------------------------------------- data


class LayoutDataset:
    def __init__(self, root: Path, limit: int | None = None):
        meta = json.loads((root / "annotations.json").read_text())
        self.root = root
        self.items = meta["images"][:limit] if limit else meta["images"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        rec = self.items[i]
        return Image.open(self.root / rec["file"]).convert("RGB"), rec, i


def collate(processor, batch):
    images, recs, ids = zip(*batch)
    annotations = [
        {"image_id": i, "annotations": [{"bbox": b, "category_id": c, "area": b[2] * b[3], "iscrowd": 0}
                                        for b, c in zip(r["boxes"], r["labels"])]}
        for r, i in zip(recs, ids)
    ]
    enc = processor(images=list(images), annotations=annotations, return_tensors="pt")
    return enc["pixel_values"], enc["labels"], recs


def loader(dataset, processor, batch_size: int, order: list[int] | None = None):
    """``order`` fixes the sample order (a per-epoch permutation), so a resumed epoch sees
    exactly the batches it has not seen yet."""
    import torch

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=order, num_workers=0,
                                       collate_fn=lambda b: collate(processor, b))


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    import torch

    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()


# --------------------------------------------------------------------------- evaluation


def load_model(path: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(path, local_files_only=True).to(device).eval()
    # map the model's labels onto DocLayNet classes (Docling's model has extra classes: dropped)
    ours = {norm_label(c): k for k, c in enumerate(CLASSES)}
    mapping = {int(i): ours.get(norm_label(name)) for i, name in model.config.id2label.items()}
    return processor, model, mapping, torch.bfloat16 if device != "cpu" else torch.float32


def evaluate_model(model_path: str, data: Path, split: str, device: str, batch_size: int, limit: int | None) -> dict:
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    processor, model, mapping, dtype = load_model(model_path, device)
    dataset = LayoutDataset(data / split, limit)
    metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", class_metrics=True, backend="pycocotools")
    size = json.loads((data / split / "annotations.json").read_text())["size"]
    t0 = time.perf_counter()
    for k in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(k, min(k + batch_size, len(dataset)))]
        images = [b[0] for b in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device != "cpu"):
            outputs = model(**inputs)
        outputs.logits, outputs.pred_boxes = outputs.logits.float(), outputs.pred_boxes.float()
        results = processor.post_process_object_detection(
            outputs, threshold=0.01, target_sizes=torch.tensor([[size, size]] * len(images)))
        preds, targets = [], []
        for res, (_, rec, _) in zip(results, batch):
            labels = [mapping.get(int(lbl)) for lbl in res["labels"]]
            keep = [i for i, lbl in enumerate(labels) if lbl is not None]
            xyxy = res["boxes"][keep].cpu()
            xywh = torch.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], dim=1) \
                if len(keep) else torch.zeros((0, 4))
            preds.append({"boxes": xywh, "scores": res["scores"][keep].cpu(),
                          "labels": torch.tensor([labels[i] for i in keep], dtype=torch.long)})
            targets.append({"boxes": torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4),
                            "labels": torch.tensor(rec["labels"], dtype=torch.long)})
        metric.update(preds, targets)
    m = metric.compute()
    per_class = {CLASSES[int(c)]: round(float(v), 4) for c, v in zip(m["classes"], m["map_per_class"])}
    return {"model": model_path, "split": split, "images": len(dataset), "seconds": round(time.perf_counter() - t0, 1),
            "map": round(float(m["map"]), 4), "map_50": round(float(m["map_50"]), 4),
            "map_75": round(float(m["map_75"]), 4), "map_per_class": per_class}


# --------------------------------------------------------------------------- training


def train(args) -> None:
    import torch
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(args.base, local_files_only=True)
    processor.size = {"height": args.size, "width": args.size}
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.base, local_files_only=True, ignore_mismatched_sizes=True, num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)},
    ).to(device)
    train_set = LayoutDataset(Path(args.data) / "train", args.limit)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr * 0.1}, {"params": rest, "lr": args.lr}],
                            weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train_set) / args.batch_size)
    warmup = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    ckpt_path, stop_flag = out / "checkpoint.pt", out / "STOP"
    best, step, start_epoch, skip = -1.0, 0, 0, 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        best, step, start_epoch, skip = ckpt["best"], ckpt["step"], ckpt["epoch"], ckpt["batch_in_epoch"]
        print(f"resumed from step {step} (epoch {start_epoch}, batch {skip})", flush=True)
    stop_flag.unlink(missing_ok=True)

    def save_checkpoint(epoch: int, batch_in_epoch: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "best": best, "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch}, tmp)
        tmp.replace(ckpt_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, losses = time.perf_counter(), []
        order = epoch_order(len(train_set), args.seed, epoch)[skip * args.batch_size:]
        batch_in_epoch = skip
        skip = 0
        try:
            for pixel_values, labels, _ in loader(train_set, processor, args.batch_size, order):
                labels = [{k: v.to(device) for k, v in lab.items()} for lab in labels]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    loss = model(pixel_values=pixel_values.to(device), labels=labels).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                opt.step()
                sched.step()
                losses.append(float(loss))
                step += 1
                batch_in_epoch += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {np.mean(losses[-50:]):.4f} "
                          f"{(time.perf_counter() - t0) / len(losses):.2f}s/it", flush=True)
                if step % args.ckpt_every == 0:
                    save_checkpoint(epoch, batch_in_epoch)
                if stop_flag.exists():
                    raise KeyboardInterrupt
        except KeyboardInterrupt:  # `stop` command or Ctrl+C: save and leave, `--resume` continues here
            save_checkpoint(epoch, batch_in_epoch)
            stop_flag.unlink(missing_ok=True)
            print(f"paused at step {step} (epoch {epoch}, batch {batch_in_epoch}); resume with --resume", flush=True)
            log.close()
            return
        save_checkpoint(epoch + 1, 0)
        model.save_pretrained(out / "last")
        processor.save_pretrained(out / "last")
        val = evaluate_model(str(out / "last"), Path(args.data), "validation", device, args.batch_size, args.val_limit)
        record = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4) if losses else None, "val_map": val["map"],
                  "val_map_50": val["map_50"], "minutes": round((time.perf_counter() - t0) / 60, 1)}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(record, flush=True)
        if val["map"] > best:
            best = val["map"]
            model.save_pretrained(out / "best")
            processor.save_pretrained(out / "best")
        save_checkpoint(epoch + 1, 0)  # remembers the new best
    log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--limit", type=int, default=None, help="pages per split (smoke tests)")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base", default=BASE_MODEL)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--size", type=int, default=640)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--val-limit", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt")
    t.add_argument("--ckpt-every", type=int, default=500, help="steps between checkpoints")
    s = sub.add_parser("stop", help="ask a running training to save a checkpoint and exit")
    s.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--out", default=None, help="write the metrics JSON here")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(Path(args.out), args.size, args.limit)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "stop":
        (Path(args.out) / "STOP").touch()
        print("stop requested: the training saves a checkpoint after the current batch and exits")
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        res = evaluate_model(args.model, Path(args.data), args.split, device, args.batch_size, args.limit)
        print(json.dumps(res, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
