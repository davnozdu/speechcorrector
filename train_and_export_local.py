#!/usr/bin/env python3
"""
Локальное дообучение на Apple Silicon (MPS) + чистый экспорт в INT8 ONNX
с исключением lm_head из квантизации (чтобы не было артефактов на редких словах и списках).
"""

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path("/Users/davnozdu/Downloads/speechcorrector")
BASE_CKPT = ROOT / "checkpoints/stage2_best"
OUT_CKPT = ROOT / "checkpoints/stage3_local_best"
ONNX_FP32_DIR = ROOT / "models/fp32_tmp"
ONNX_INT8_DIR = ROOT / "models/sage_ru_corrector_int8"


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return r["source"], r["target"]


def train_local_mps(epochs: int = 3, batch_size: int = 16, grad_accum: int = 2, lr: float = 5e-5):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== LOCAL TRAINING ON {device.upper()} ===", flush=True)

    rng = random.Random(2026)
    synth = [json.loads(l) for l in (ROOT / "data/dictation_synth.jsonl").open(encoding="utf-8") if l.strip()]
    books = [json.loads(l) for l in (ROOT / "data/train.jsonl").open(encoding="utf-8") if l.strip()]
    rng.shuffle(books)
    mix = synth + books[:2000]
    rng.shuffle(mix)
    print(f"Training rows: {len(mix)} (synth={len(synth)}, books=2000) | epochs={epochs} lr={lr}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_CKPT, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_CKPT).to(device)

    ds = PairDataset(mix)

    def collate_fn(batch):
        srcs, tgts = zip(*batch)
        enc = tokenizer(list(srcs), padding=True, truncation=True, max_length=192, return_tensors="pt")
        lab = tokenizer(text_target=list(tgts), padding=True, truncation=True, max_length=192, return_tensors="pt")["input_ids"]
        lab[lab == tokenizer.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / grad_accum) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.08 * total_steps), total_steps)

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum

            if step % grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0 or step == len(loader):
                print(
                    f"Local Ep {ep}/{epochs} | step {step}/{len(loader)} | loss={running_loss/step:.4f} | {time.time()-t0:.0f}s",
                    flush=True,
                )

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(OUT_CKPT)
    tokenizer.save_pretrained(OUT_CKPT)
    print(f"Saved polished checkpoint to {OUT_CKPT}", flush=True)


def export_clean_onnx():
    print("\n=== EXPORTING CLEAN INT8 ONNX (KEEPING lm_head IN FP32) ===", flush=True)
    ONNX_FP32_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(OUT_CKPT, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUT_CKPT).cpu().eval()
    model.config.use_cache = False

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

    enc_wrap = EncoderWrapper(model.encoder).eval()
    dec_wrap = DecoderWrapper(model).eval()

    dummy_enc_ids = torch.randint(0, 1000, (1, 16), dtype=torch.long)
    dummy_enc_mask = torch.ones((1, 16), dtype=torch.long)
    dummy_dec_ids = torch.tensor([[model.config.decoder_start_token_id, 10, 20]], dtype=torch.long)
    with torch.no_grad():
        dummy_hidden = enc_wrap(dummy_enc_ids, dummy_enc_mask)

    enc_fp32 = ONNX_FP32_DIR / "encoder_model.onnx"
    dec_fp32 = ONNX_FP32_DIR / "decoder_model.onnx"

    torch.onnx.export(
        enc_wrap,
        (dummy_enc_ids, dummy_enc_mask),
        str(enc_fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "src_len"}, "attention_mask": {0: "batch", 1: "src_len"}, "last_hidden_state": {0: "batch", 1: "src_len"}},
        opset_version=14,
        dynamo=False,
    )
    torch.onnx.export(
        dec_wrap,
        (dummy_dec_ids, dummy_hidden, dummy_enc_mask),
        str(dec_fp32),
        input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch", 1: "tgt_len"},
            "encoder_hidden_states": {0: "batch", 1: "src_len"},
            "encoder_attention_mask": {0: "batch", 1: "src_len"},
            "logits": {0: "batch", 1: "tgt_len"},
        },
        opset_version=14,
        dynamo=False,
    )

    # Находим ноду lm_head, чтобы оставить финальную проекцию словаря в FP32 (без потери точности)
    dec_proto = onnx.load(str(dec_fp32))
    exclude_nodes = [n.name for n in dec_proto.graph.node if "lm_head" in n.name.lower()]
    print(f"Excluding from INT8 quantization (kept FP32 for accuracy): {exclude_nodes}", flush=True)

    enc_int8 = ONNX_INT8_DIR / "encoder_model.int8.onnx"
    dec_int8 = ONNX_INT8_DIR / "decoder_model.int8.onnx"

    quantize_dynamic(str(enc_fp32), str(enc_int8), weight_type=QuantType.QUInt8)
    quantize_dynamic(str(dec_fp32), str(dec_int8), weight_type=QuantType.QUInt8, nodes_to_exclude=exclude_nodes)

    tokenizer.save_pretrained(ONNX_INT8_DIR)
    for fname in ("config.json", "generation_config.json"):
        src = OUT_CKPT / fname
        if src.exists():
            shutil.copy2(src, ONNX_INT8_DIR / fname)

    # Удаляем временные FP32 файлы
    shutil.rmtree(ONNX_FP32_DIR, ignore_errors=True)

    total_mb = sum(p.stat().st_size for p in ONNX_INT8_DIR.iterdir()) / 1e6
    print(f"Exported final ONNX model to {ONNX_INT8_DIR} ({total_mb:.1f} MB total)", flush=True)
    for p in sorted(ONNX_INT8_DIR.iterdir()):
        print(f"  {p.stat().st_size/1e6:6.1f} MB  {p.name}")


def benchmark_local():
    tok = Tokenizer.from_file(str(ONNX_INT8_DIR / "tokenizer.json"))
    cfg = json.loads((ONNX_INT8_DIR / "config.json").read_text(encoding="utf-8"))
    start_id = int(cfg.get("decoder_start_token_id", 0))
    eos_id = int(cfg.get("eos_token_id", 2))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession(str(ONNX_INT8_DIR / "encoder_model.int8.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(ONNX_INT8_DIR / "decoder_model.int8.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

    def correct(text: str, max_tokens: int = 128):
        t0 = time.time()
        ids = np.array([tok.encode(text).ids], dtype=np.int64)
        mask = np.ones_like(ids, dtype=np.int64)
        hidden = enc_sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        dec_ids = [start_id]
        for _ in range(max_tokens):
            logits = dec_sess.run(
                None,
                {
                    "decoder_input_ids": np.array([dec_ids], dtype=np.int64),
                    "encoder_hidden_states": hidden,
                    "encoder_attention_mask": mask,
                },
            )[0]
            next_id = int(np.argmax(logits[0, -1]))
            if next_id == eos_id:
                break
            dec_ids.append(next_id)
        return tok.decode(dec_ids[1:], skip_special_tokens=True).strip(), (time.time() - t0) * 1000.0

    correct("привет")

    tests = [
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

    times = []
    print("\n=== FINAL LOCAL M2 PRO ONNX BENCHMARK ===", flush=True)
    for t in tests:
        out, ms = correct(t)
        times.append(ms)
        print(f"  [{ms:5.0f} ms] IN : {t}")
        print(f"             OUT: {out}", flush=True)
    print(f"\nAverage latency: {sum(times)/len(times):.0f} ms | Min: {min(times):.0f} ms | Max: {max(times):.0f} ms", flush=True)


if __name__ == "__main__":
    train_local_mps(epochs=3, batch_size=16, grad_accum=2, lr=5e-5)
    export_clean_onnx()
    benchmark_local()
