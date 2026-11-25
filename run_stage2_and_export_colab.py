#!/usr/bin/env python3
"""
Stage 2: Дообучение поверх checkpoints/best (Epoch 1, CER=1.96%) на смеси:
  - 5000 примеров современной диктовки (опечатки, оговорки, окончания, DeepSeek/GitHub и др.)
  - 3000 книжных примеров (анти-забывание синтаксиса)
Валидация на 500 отложенных современных примерах + 500 книжных примерах,
затем экспорт лучшей модели в INT8 ONNX и упаковка в архив.
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

workdir = Path("/content/speechcorrector")
os.chdir(workdir)
sys.path.insert(0, str(workdir))
from colab_benchmark_and_train import CorrectionDataset, compute_metrics, generate_batch

# Копируем dictation_synth.jsonl из корня (куда кладёт colab upload)
for base in ("/", "/content", "/root", str(workdir)):
    src = Path(base) / "dictation_synth.jsonl"
    if src.exists():
        dst = workdir / "data" / "dictation_synth.jsonl"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        print(f"Ready: {dst}", flush=True)
        break

rng = random.Random(777)
synth = [json.loads(l) for l in (workdir / "data/dictation_synth.jsonl").open(encoding="utf-8") if l.strip()]
rng.shuffle(synth)
synth_val = synth[:400]
synth_train = synth[400:]

books = [json.loads(l) for l in (workdir / "data/train.jsonl").open(encoding="utf-8") if l.strip()]
rng.shuffle(books)
book_val = [json.loads(l) for l in (workdir / "data/val.jsonl").open(encoding="utf-8") if l.strip()][:500]

mix_train = synth_train + books[:3000]
rng.shuffle(mix_train)
print(f"Stage 2 train: {len(mix_train)} rows | synth_val: {len(synth_val)} | book_val: {len(book_val)}", flush=True)

device = "cuda"
base_dir = workdir / "checkpoints/best"
tokenizer = AutoTokenizer.from_pretrained(base_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(base_dir).to(device)

# Baseline on synth_val BEFORE Stage 2
preds0 = generate_batch(model, tokenizer, [r["source"] for r in synth_val], device, prefix="", batch_size=32)
m0 = compute_metrics(preds0, [r["target"] for r in synth_val], [r["mode"] for r in synth_val])
print("\n--- Modern Dictation Val BEFORE Stage 2 ---")
print(json.dumps(m0, ensure_ascii=False, indent=2), flush=True)

ds = CorrectionDataset(mix_train, prefix="")

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    enc = tokenizer(list(srcs), padding=True, truncation=True, max_length=256, return_tensors="pt")
    lab = tokenizer(text_target=list(tgts), padding=True, truncation=True, max_length=256, return_tensors="pt")["input_ids"]
    lab[lab == tokenizer.pad_token_id] = -100
    enc["labels"] = lab
    return enc

batch_size = 32
grad_accum = 2
epochs = 2
lr = 3e-5

loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = math.ceil(len(loader) / grad_accum) * epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.08 * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=True)

out_dir = workdir / "checkpoints_stage2/best"
best_score = 1e9

for ep in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss / grad_accum
        scaler.scale(loss).backward()
        running_loss += loss.item() * grad_accum
        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        if step % 100 == 0 or step == len(loader):
            print(f"Stage2 Ep {ep}/{epochs} | step {step}/{len(loader)} | loss={running_loss/step:.4f} | {time.time()-t0:.0f}s", flush=True)

    ps = generate_batch(model, tokenizer, [r["source"] for r in synth_val], device, prefix="", batch_size=32)
    ms = compute_metrics(ps, [r["target"] for r in synth_val], [r["mode"] for r in synth_val])
    pb = generate_batch(model, tokenizer, [r["source"] for r in book_val], device, prefix="", batch_size=32)
    mb = compute_metrics(pb, [r["target"] for r in book_val], [r["mode"] for r in book_val])
    print(f"\n--- Stage 2 Epoch {ep} | Modern Dictation Val ---", flush=True)
    print(json.dumps(ms, ensure_ascii=False, indent=2), flush=True)
    print(f"--- Stage 2 Epoch {ep} | Book Val (anti-forgetting) ---", flush=True)
    print(json.dumps({"exact_%": mb["exact_%"], "cer_%": mb["cer_%"], "punct_f1_%": mb["punct_f1_%"]}, ensure_ascii=False), flush=True)

    # Комбинированный критерий: современный CER + книжный CER
    score = ms["cer_%"] + 0.5 * mb["cer_%"]
    if score < best_score:
        best_score = score
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "corrector_meta.json").write_text(
            json.dumps({"prefix": "", "modern_val": ms, "book_val": mb}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"--> Saved best Stage 2 checkpoint (combined score={score:.2f})", flush=True)

# Проверка на живых тестовых фразах
best_mdl = AutoModelForSeq2SeqLM.from_pretrained(out_dir).to(device).eval()
live_tests = [
    "привет подскажи пожалуста во сколько мы сигодня сазвонимся",
    "давай встретимся завтра в пять ой нет в шесть вечера",
    "слушай я я хотел сказать что задача уже готова и можно проверять на гитхаб",
    "очень хорошо помогает дипсик 4 1 флэш когда нужно заранее создать тексты и обучить модель",
    "я залил модель в гугл колаб и сравнил дип сик с джемини через апи",
    "нужно экспортировать модель из пайторч в оннкс рантайм для быстрой работы на макос",
    "ксожалению билетов на утренний рейс уже нет придется лететь вечером",
    "главное не торопится и внимательно проверить все цифры перед отправкой",
    "в прошлый году мы запустили большой обновление на главный странице",
    "нужно исправить неправильный окончания в распознанный тексте",
    "Проверь, пожалуйста, работает ли горячая клавиша в Chrome.",
]
enc = tokenizer(live_tests, padding=True, return_tensors="pt").to(device)
with torch.inference_mode():
    out = best_mdl.generate(**enc, max_new_tokens=160, num_beams=1)
print("\n=== LIVE DICTATION TESTS ===", flush=True)
for inp, pred in zip(live_tests, tokenizer.batch_decode(out, skip_special_tokens=True)):
    print(f"  IN : {inp}")
    print(f"  OUT: {pred.strip()}")

# Экспорт в INT8 ONNX
print("\nInstalling optimum[onnxruntime] for ONNX export...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optimum[onnxruntime]", "onnx"])

cmd_export = [
    sys.executable,
    "colab_benchmark_and_train.py",
    "--mode", "export",
    "--model-dir", str(out_dir),
    "--out-dir", "onnx_out",
]
p = subprocess.Popen(cmd_export, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
if p.wait() != 0:
    raise RuntimeError("ONNX export failed")

archive_path = Path("/speechcorr_onnx_int8.tar.gz")
with tarfile.open(archive_path, "w:gz") as tf:
    tf.add(workdir / "onnx_out/int8", arcname="sage_ru_corrector_int8")
print(f"\nPacked archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)", flush=True)
