#!/usr/bin/env python3
"""
Локальный тест и замер скорости экспортированной INT8 ONNX модели на Apple M2 Pro.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path("/Users/davnozdu/Downloads/speechcorrector/models/sage_ru_corrector_int8")


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    start_id = int(cfg.get("decoder_start_token_id", 0))
    eos_id = int(cfg.get("eos_token_id", 2))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t_load = time.time()
    enc_sess = ort.InferenceSession(str(MODEL_DIR / "encoder_model.int8.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(MODEL_DIR / "decoder_model.int8.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX models in {(time.time() - t_load)*1000:.0f} ms")

    def correct(text: str, max_tokens: int = 128) -> tuple[str, float]:
        t0 = time.time()
        encoding = tok.encode(text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.ones_like(ids, dtype=np.int64)
        hidden = enc_sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]

        dec_ids = [start_id]
        for _ in range(max_tokens):
            d_in = np.array([dec_ids], dtype=np.int64)
            logits = dec_sess.run(
                None,
                {
                    "decoder_input_ids": d_in,
                    "encoder_hidden_states": hidden,
                    "encoder_attention_mask": mask,
                },
            )[0]
            next_id = int(np.argmax(logits[0, -1]))
            if next_id == eos_id:
                break
            dec_ids.append(next_id)

        out = tok.decode(dec_ids[1:], skip_special_tokens=True).strip()
        return out, (time.time() - t0) * 1000.0

    # Warmup
    correct("привет как дела")

    tests = [
        "привет подскажи пожалуста во сколько мы сигодня сазвонимся",
        "слушай я я хотел сказать что задача уже готова и можно проверять на гитхаб",
        "очень хорошо помогает дипсик 4 1 флэш когда нужно заранее создать тексты и обучить модель",
        "я залил модель в гугл колаб и сравнил дип сик с джемини через апи",
        "нужно экспортировать модель из пайторч в оннкс рантайм для быстрой работы на макос",
        "в прошлый году мы запустили большой обновление на главный странице",
        "нужно исправить неправильный окончания в распознанный тексте",
        "не забудь купить по дороге домой молоко хлеб яйца и свежие помидоры",
        "ксожалению билетов на утренний рейс уже нет придется лететь вечером",
        "пьер не слышал что они говорили но знал что говорят про него он покраснел и отвернулся",
        "главное не торопится и внимательно проверить все цифры перед отправкой",
        "Проверь, пожалуйста, работает ли горячая клавиша в Chrome.",
    ]

    times = []
    print("\n=== LOCAL APPLE M2 PRO ONNX BENCHMARK ===")
    for text in tests:
        out, ms = correct(text)
        times.append(ms)
        print(f"  [{ms:5.0f} ms] IN : {text}")
        print(f"             OUT: {out}")
    print(f"\nAverage latency: {sum(times)/len(times):.0f} ms | Min: {min(times):.0f} ms | Max: {max(times):.0f} ms")


if __name__ == "__main__":
    main()
