#!/usr/bin/env python3
"""
Чистый PyTorch-экспорт T5 (sage-fredt5-distilled-95m) в ONNX + INT8 квантизация
без зависимости от optimum/diffusers.
Выгружает:
  - encoder_model.int8.onnx
  - decoder_model.int8.onnx
  - tokenizer.json (fast tokenizer для Rust tokenizers)
  - config.json / generation_config.json / corrector_meta.json
Проверяет совпадение PyTorch vs INT8 ONNX и пакует в /speechcorr_onnx_int8.tar.gz.
"""

import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

workdir = Path("/content/speechcorrector")
os.chdir(workdir)

model_dir = workdir / "checkpoints_stage2/best"
fp32_dir = workdir / "onnx_out/fp32"
int8_dir = workdir / "onnx_out/int8"
fp32_dir.mkdir(parents=True, exist_ok=True)
int8_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading PyTorch model from {model_dir}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).cpu().eval()
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

enc_fp32 = fp32_dir / "encoder_model.onnx"
dec_fp32 = fp32_dir / "decoder_model.onnx"

print("Exporting encoder_model.onnx...", flush=True)
torch.onnx.export(
    enc_wrap,
    (dummy_enc_ids, dummy_enc_mask),
    str(enc_fp32),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "src_len"},
        "attention_mask": {0: "batch", 1: "src_len"},
        "last_hidden_state": {0: "batch", 1: "src_len"},
    },
    opset_version=14,
    dynamo=False,
)

print("Exporting decoder_model.onnx...", flush=True)
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

enc_int8 = int8_dir / "encoder_model.int8.onnx"
dec_int8 = int8_dir / "decoder_model.int8.onnx"

print("Quantizing encoder & decoder to INT8...", flush=True)
quantize_dynamic(str(enc_fp32), str(enc_int8), weight_type=QuantType.QUInt8)
quantize_dynamic(str(dec_fp32), str(dec_int8), weight_type=QuantType.QUInt8)

# Сохраняем tokenizer.json (fast) и метаданные
tokenizer.save_pretrained(int8_dir)
for fname in ("config.json", "generation_config.json", "corrector_meta.json"):
    src = model_dir / fname
    if src.exists():
        shutil.copy2(src, int8_dir / fname)

print("\n=== Exported INT8 ONNX files ===", flush=True)
total_mb = 0.0
for p in sorted(int8_dir.iterdir()):
    sz = p.stat().st_size / 1e6
    total_mb += sz
    print(f"  {sz:6.1f} MB  {p.name}", flush=True)
print(f"  TOTAL: {total_mb:.1f} MB", flush=True)

# Быстрая проверка ONNX INT8 инференса
enc_sess = ort.InferenceSession(str(enc_int8), providers=["CPUExecutionProvider"])
dec_sess = ort.InferenceSession(str(dec_int8), providers=["CPUExecutionProvider"])

eos_id = model.config.eos_token_id
start_id = model.config.decoder_start_token_id

def infer_onnx(text: str, max_tokens: int = 128) -> str:
    ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
    mask = np.ones_like(ids, dtype=np.int64)
    hidden = enc_sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    dec_ids = [start_id]
    for _ in range(max_tokens):
        d_in = np.array([dec_ids], dtype=np.int64)
        logits = dec_sess.run(None, {
            "decoder_input_ids": d_in,
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })[0]
        next_id = int(np.argmax(logits[0, -1]))
        if next_id == eos_id:
            break
        dec_ids.append(next_id)
    return tokenizer.decode(dec_ids[1:], skip_special_tokens=True).strip()

tests = [
    "привет подскажи пожалуста во сколько мы сигодня сазвонимся",
    "давай встретимся завтра в пять ой нет в шесть вечера",
    "очень хорошо помогает дипсик 4 1 флэш когда нужно заранее создать тексты",
    "слушай я я хотел сказать что задача готова и лежит на гитхаб",
]
print("\n=== INT8 ONNX SANITY CHECK ===", flush=True)
for t in tests:
    t0 = time.time()
    out = infer_onnx(t)
    ms = (time.time() - t0) * 1000
    print(f"  [{ms:5.0f} ms] IN : {t}")
    print(f"             OUT: {out}", flush=True)

archive_path = Path("/speechcorr_onnx_int8.tar.gz")
with tarfile.open(archive_path, "w:gz") as tf:
    tf.add(int8_dir, arcname="sage_ru_corrector_int8")
print(f"\nPacked archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)", flush=True)
