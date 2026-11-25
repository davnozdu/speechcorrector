#!/usr/bin/env python3
"""
Benchmark, fine-tuning, and ONNX export for Russian dictation corrector
based on ai-forever/sage-fredt5-distilled-95m.

Run on Google Colab (T4/L4 GPU):
  python3 colab_benchmark_and_train.py --mode benchmark --val data/val.jsonl --limit 500
  python3 colab_benchmark_and_train.py --mode train --train data/train.jsonl --val data/val.jsonl --epochs 3
  python3 colab_benchmark_and_train.py --mode export --model-dir checkpoints/best --out-dir onnx_out
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

MODEL_ID = "ai-forever/sage-fredt5-distilled-95m"
PUNCT_CHARS = set(".,?!:;—…")


def load_jsonl(path: Path, limit: int = 0) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def compute_metrics(preds: List[str], golds: List[str], modes: List[str]) -> Dict:
    total = len(preds)
    exact = 0
    cer_num = cer_den = 0
    wer_num = wer_den = 0
    punct_tp = punct_fp = punct_fn = 0

    by_mode = defaultdict(lambda: {"n": 0, "exact": 0, "cer_num": 0, "cer_den": 0})

    for pred, gold, mode in zip(preds, golds, modes):
        p = pred.strip()
        g = gold.strip()
        is_ex = int(p == g)
        exact += is_ex

        c_dist = levenshtein(p, g)
        cer_num += c_dist
        cer_den += max(len(g), 1)

        pw, gw = p.split(), g.split()
        wer_num += levenshtein(pw, gw)
        wer_den += max(len(gw), 1)

        # Punctuation multiset F1
        pc = Counter(ch for ch in p if ch in PUNCT_CHARS)
        gc = Counter(ch for ch in g if ch in PUNCT_CHARS)
        tp = sum((pc & gc).values())
        punct_tp += tp
        punct_fp += sum(pc.values()) - tp
        punct_fn += sum(gc.values()) - tp

        m = by_mode[mode]
        m["n"] += 1
        m["exact"] += is_ex
        m["cer_num"] += c_dist
        m["cer_den"] += max(len(g), 1)

    prec = punct_tp / max(punct_tp + punct_fp, 1)
    rec = punct_tp / max(punct_tp + punct_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    mode_summary = {}
    for mode, m in sorted(by_mode.items()):
        mode_summary[mode] = {
            "n": m["n"],
            "exact_%": round(100.0 * m["exact"] / max(m["n"], 1), 2),
            "cer_%": round(100.0 * m["cer_num"] / max(m["cer_den"], 1), 2),
        }

    return {
        "n": total,
        "exact_%": round(100.0 * exact / max(total, 1), 2),
        "cer_%": round(100.0 * cer_num / max(cer_den, 1), 2),
        "wer_%": round(100.0 * wer_num / max(wer_den, 1), 2),
        "punct_p_%": round(100.0 * prec, 2),
        "punct_r_%": round(100.0 * rec, 2),
        "punct_f1_%": round(100.0 * f1, 2),
        "by_mode": mode_summary,
    }


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    prefix: str = "",
    batch_size: int = 32,
    max_new_tokens: int = 160,
) -> List[str]:
    model.eval()
    preds: List[str] = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(d.strip() for d in decoded)
    return preds


def detect_best_prefix(model, tokenizer, sample_rows: List[dict], device: str) -> str:
    """Проверяем эмпирически, нужен ли префикс <LM> для этой модели."""
    srcs = [r["source"] for r in sample_rows[:24]]
    golds = [r["target"] for r in sample_rows[:24]]
    modes = [r["mode"] for r in sample_rows[:24]]
    best_prefix = ""
    best_cer = 1e9
    for pref in ("", "<LM>"):
        preds = generate_batch(model, tokenizer, srcs, device, prefix=pref, batch_size=24)
        m = compute_metrics(preds, golds, modes)
        print(f"[prefix check] prefix={pref!r:6s} -> CER={m['cer_%']}% | PunctF1={m['punct_f1_%']}% | Exact={m['exact_%']}%")
        print(f"   sample IN : {srcs[0]}")
        print(f"   sample OUT: {preds[0]}")
        if m["cer_%"] < best_cer:
            best_cer = m["cer_%"]
            best_prefix = pref
    print(f"--> Selected prefix: {best_prefix!r}")
    return best_prefix


class CorrectionDataset(Dataset):
    def __init__(self, rows: List[dict], prefix: str = ""):
        self.rows = rows
        self.prefix = prefix

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return self.prefix + r["source"], r["target"]


def run_benchmark(model_path: str, val_path: Path, limit: int, batch_size: int, prefix: str = "auto"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    val_rows = load_jsonl(val_path, limit=limit)
    if prefix == "auto":
        prefix = detect_best_prefix(model, tokenizer, val_rows, device)

    t0 = time.time()
    preds = generate_batch(
        model,
        tokenizer,
        [r["source"] for r in val_rows],
        device,
        prefix=prefix,
        batch_size=batch_size,
    )
    dt = time.time() - t0

    golds = [r["target"] for r in val_rows]
    modes = [r["mode"] for r in val_rows]
    metrics = compute_metrics(preds, golds, modes)
    metrics["prefix"] = prefix
    metrics["sec_total"] = round(dt, 2)
    metrics["ms_per_sample"] = round(1000.0 * dt / max(len(val_rows), 1), 2)

    print("\n=== BENCHMARK RESULTS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== SAMPLE PREDICTIONS ===")
    shown = 0
    for r, p in zip(val_rows, preds):
        if p != r["target"] and shown < 8:
            print(f"[{r['mode']}]")
            print(f"  IN  : {r['source']}")
            print(f"  PRED: {p}")
            print(f"  GOLD: {r['target']}")
            shown += 1

    return metrics, prefix


def run_train(
    model_path: str,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    val_limit: int,
    prefix: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {model_path} on {device} | epochs={epochs} bs={batch_size}x{grad_accum} lr={lr}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path, limit=val_limit)

    if prefix == "auto":
        prefix = detect_best_prefix(model, tokenizer, val_rows, device)

    # Baseline eval before training
    print("\n--- Baseline (Epoch 0) ---")
    base_preds = generate_batch(
        model, tokenizer, [r["source"] for r in val_rows], device, prefix=prefix, batch_size=32
    )
    base_m = compute_metrics(base_preds, [r["target"] for r in val_rows], [r["mode"] for r in val_rows])
    print(json.dumps(base_m, ensure_ascii=False, indent=2))

    ds = CorrectionDataset(train_rows, prefix=prefix)

    def collate_fn(batch):
        srcs, tgts = zip(*batch)
        enc = tokenizer(
            list(srcs),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        lab = tokenizer(
            text_target=list(tgts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )["input_ids"]
        lab[lab == tokenizer.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / grad_accum) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    best_cer = base_m["cer_%"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best"

    # Save prefix metadata
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                loss = model(**batch).loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * grad_accum

            if step % grad_accum == 0 or step == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 250 == 0 or step == len(loader):
                avg_loss = running_loss / step
                print(f"Epoch {ep}/{epochs} | step {step}/{len(loader)} | loss={avg_loss:.4f} | elapsed={time.time()-t0:.0f}s", flush=True)

        # Eval
        preds = generate_batch(
            model, tokenizer, [r["source"] for r in val_rows], device, prefix=prefix, batch_size=32
        )
        m = compute_metrics(preds, [r["target"] for r in val_rows], [r["mode"] for r in val_rows])
        print(f"\n--- Validation after Epoch {ep} ---")
        print(json.dumps(m, ensure_ascii=False, indent=2), flush=True)

        if m["cer_%"] < best_cer:
            best_cer = m["cer_%"]
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "corrector_meta.json").write_text(
                json.dumps({"prefix": prefix, "val_metrics": m, "base_metrics": base_m}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"--> Saved new best checkpoint to {best_dir} (CER={best_cer}%)", flush=True)

    print("\n=== SAMPLE PREDICTIONS (BEST CHECKPOINT) ===")
    if best_dir.exists():
        best_model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
        preds = generate_batch(
            best_model, tokenizer, [r["source"] for r in val_rows[:12]], device, prefix=prefix, batch_size=12
        )
        for r, p in zip(val_rows[:12], preds):
            print(f"[{r['mode']}]")
            print(f"  IN  : {r['source']}")
            print(f"  PRED: {p}")
            print(f"  GOLD: {r['target']}")


def run_export(model_dir: Path, out_dir: Path):
    """Экспорт в ONNX (FP32 + dynamic INT8) через optimum."""
    import subprocess
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_dir = out_dir / "fp32"
    int8_dir = out_dir / "int8"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(model_dir),
        "--task",
        "text2text-generation-with-past",
        str(fp32_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    # Quantize all .onnx files to dynamic INT8
    for onnx_file in sorted(fp32_dir.glob("*.onnx")):
        dst = int8_dir / onnx_file.name
        print(f"Quantizing {onnx_file.name} -> {dst.name}...", flush=True)
        quantize_dynamic(
            model_input=str(onnx_file),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
        )

    # Copy tokenizer & config files to int8_dir
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json", "corrector_meta.json"):
        src = fp32_dir / fname
        if not src.exists():
            src = model_dir / fname
        if src.exists():
            (int8_dir / fname).write_bytes(src.read_bytes())

    print("\nExported INT8 ONNX files:")
    for p in sorted(int8_dir.iterdir()):
        print(f"  {p.stat().st_size / 1e6:6.1f} MB  {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark", "train", "export"], required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--model-dir", type=Path, default=Path("checkpoints/best"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--prefix", default="auto")
    args = ap.parse_args()

    if args.mode == "benchmark":
        run_benchmark(args.model, args.val, args.limit, args.batch_size, args.prefix)
    elif args.mode == "train":
        run_train(
            args.model,
            args.train,
            args.val,
            args.out_dir,
            args.epochs,
            args.batch_size,
            args.grad_accum,
            args.lr,
            args.limit,
            args.prefix,
        )
    elif args.mode == "export":
        run_export(args.model_dir, args.out_dir)


if __name__ == "__main__":
    main()
