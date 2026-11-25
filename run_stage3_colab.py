#!/usr/bin/env python3
"""
Единое совместное дообучение ai-forever/sage-fredt5-distilled-95m на Colab T4 GPU:
  - 20 000 книжных отрывков (чистый русский синтаксис + пунктуация + регистр)
  - 7 000 современных пар (DeepSeek 4.1 Flash, IT-бренды, окончания, списки, оговорки, ASR-ошибки)
  - 2 эпохи (cosine LR 8e-5) -> без переобучения и без забывания
  - Чистый экспорт в INT8 ONNX с сохранением lm_head в FP32 (без артефактов квантизации)
  - Упаковка в /speechcorr_final_onnx.tar.gz
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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

workdir = Path("/content/stage3")
workdir.mkdir(parents=True, exist_ok=True)
os.chdir(workdir)

for base in ("/", "/content", "/root", str(workdir)):
    src = Path(base) / "data_all.tar.gz"
    if src.exists():
        with tarfile.open(src, "r:gz") as tf:
            tf.extractall(".")
        print("Extracted data_all.tar.gz", flush=True)
        break

MODEL_ID = "ai-forever/sage-fredt5-distilled-95m"
OUT_CKPT = workdir / "checkpoints/final_best"
ONNX_FP32_DIR = workdir / "onnx_fp32"
ONNX_INT8_DIR = workdir / "onnx_int8"


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return r["source"], r["target"]


rng = random.Random(2026)
synth = [json.loads(l) for l in (workdir / "data/dictation_synth.jsonl").open(encoding="utf-8") if l.strip()]
books = [json.loads(l) for l in (workdir / "data/train.jsonl").open(encoding="utf-8") if l.strip()]
rng.shuffle(books)
# Даём современным парам (бренды, окончания, списки, оговорки) вес x2 в совместной смеси
mix = synth * 2 + books[:20000]
rng.shuffle(mix)
print(f"Unified train: {len(mix)} rows (synth={len(synth)}x2, books=20000)", flush=True)

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)

ds = PairDataset(mix)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    enc = tokenizer(list(srcs), padding=True, truncation=True, max_length=192, return_tensors="pt")
    lab = tokenizer(text_target=list(tgts), padding=True, truncation=True, max_length=192, return_tensors="pt")["input_ids"]
    lab[lab == tokenizer.pad_token_id] = -100
    enc["labels"] = lab
    return enc

batch_size = 32
grad_accum = 2
epochs = 2
lr = 8e-5

loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = math.ceil(len(loader) / grad_accum) * epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=True)

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
        if step % 250 == 0 or step == len(loader):
            print(f"Ep {ep}/{epochs} | step {step}/{len(loader)} | loss={running_loss/step:.4f} | {time.time()-t0:.0f}s", flush=True)

OUT_CKPT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_CKPT)
tokenizer.save_pretrained(OUT_CKPT)

# Live PyTorch check
model.eval()
live_tests = [
    "привет подскажи пожалуста во сколько мы сигодня сазвонимся",
    "давай встретимся завтра в пять ой нет в шесть вечера",
    "слушай я я хотел сказать что задача уже готова и можно проверять на гитхаб",
    "очень хорошо помогает дипсик 4 1 флэш когда нужно заранее создать тексты и обучить модель",
    "я залил модель в гугл колаб и сравнил дип сик с джемини через апи",
    "нужно экспортировать модель из пайторч в оннкс рантайм для быстрой работы на макос",
    "в прошлый году мы запустили большой обновление на главный странице",
    "нужно исправить неправильный окончания в распознанный тексте",
    "не забудь купить по дороге домой молоко хлеб яйца и свежие помидоры",
    "ксожалению билетов на утренний рейс уже нет придется лететь вечером",
    "главное не торопится и внимательно проверить все цифры перед отправкой",
    "Проверь, пожалуйста, работает ли горячая клавиша в Chrome.",
]
enc = tokenizer(live_tests, padding=True, return_tensors="pt").to(device)
with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=160, num_beams=1)
print("\n=== FINAL PYTORCH LIVE TESTS ===", flush=True)
for inp, pred in zip(live_tests, tokenizer.batch_decode(out, skip_special_tokens=True)):
    print(f"  IN : {inp}")
    print(f"  OUT: {pred.strip()}", flush=True)

# Clean ONNX export (keeping lm_head in FP32)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"])
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

ONNX_FP32_DIR.mkdir(parents=True, exist_ok=True)
ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)

model_cpu = AutoModelForSeq2SeqLM.from_pretrained(OUT_CKPT).cpu().eval()
model_cpu.config.use_cache = False

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

class DecoderWrapper(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.decoder = full_model.decoder
        self.lm_head = full_model.lm_head
        self.d_model = full_model.config.d_model
        self.tie_word_embeddings = bool(getattr(full_model.config, "tie_word_embeddings", False))
    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        dec_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]
        if self.tie_word_embeddings:
            dec_out = dec_out * (self.d_model ** -0.5)
        return self.lm_head(dec_out)

enc_wrap = EncoderWrapper(model_cpu.encoder).eval()
dec_wrap = DecoderWrapper(model_cpu).eval()
dummy_enc_ids = torch.randint(0, 1000, (1, 16), dtype=torch.long)
dummy_enc_mask = torch.ones((1, 16), dtype=torch.long)
dummy_dec_ids = torch.tensor([[model_cpu.config.decoder_start_token_id, 10, 20]], dtype=torch.long)
with torch.no_grad():
    dummy_hidden = enc_wrap(dummy_enc_ids, dummy_enc_mask)

enc_fp32 = ONNX_FP32_DIR / "encoder_model.onnx"
dec_fp32 = ONNX_FP32_DIR / "decoder_model.onnx"
torch.onnx.export(
    enc_wrap, (dummy_enc_ids, dummy_enc_mask), str(enc_fp32),
    input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch", 1: "src_len"}, "attention_mask": {0: "batch", 1: "src_len"}, "last_hidden_state": {0: "batch", 1: "src_len"}},
    opset_version=14, dynamo=False,
)
torch.onnx.export(
    dec_wrap, (dummy_dec_ids, dummy_hidden, dummy_enc_mask), str(dec_fp32),
    input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"], output_names=["logits"],
    dynamic_axes={"decoder_input_ids": {0: "batch", 1: "tgt_len"}, "encoder_hidden_states": {0: "batch", 1: "src_len"}, "encoder_attention_mask": {0: "batch", 1: "src_len"}, "logits": {0: "batch", 1: "tgt_len"}},
    opset_version=14, dynamo=False,
)

dec_proto = onnx.load(str(dec_fp32))
exclude_nodes = [n.name for n in dec_proto.graph.node if "lm_head" in n.name.lower()]
print("Excluding from INT8 (kept FP32):", exclude_nodes, flush=True)

enc_int8 = ONNX_INT8_DIR / "encoder_model.int8.onnx"
dec_int8 = ONNX_INT8_DIR / "decoder_model.int8.onnx"
quantize_dynamic(str(enc_fp32), str(enc_int8), weight_type=QuantType.QUInt8)
quantize_dynamic(str(dec_fp32), str(dec_int8), weight_type=QuantType.QUInt8, nodes_to_exclude=exclude_nodes)

tokenizer.save_pretrained(ONNX_INT8_DIR)
for fname in ("config.json", "generation_config.json"):
    src = OUT_CKPT / fname
    if src.exists():
        shutil.copy2(src, ONNX_INT8_DIR / fname)

archive_path = Path("/speechcorr_final_onnx.tar.gz")
with tarfile.open(archive_path, "w:gz") as tf:
    tf.add(ONNX_INT8_DIR, arcname="sage_ru_corrector_int8")
print(f"Packed final archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)", flush=True)
