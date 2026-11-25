# speechcorrector

Лёгкая локальная ONNX-модель для постобработки русской диктовки после ASR (например, **GigaAM v3** в [LLM-dict](https://github.com/davnozdu/LLM-dict)).

Основана на [`ai-forever/sage-fredt5-distilled-95m`](https://huggingface.co/ai-forever/sage-fredt5-distilled-95m) (95M параметров), дообучена на коротких русских отрывках и современных фразах диктовки, экспортирована в **ONNX** (веса трансформера INT8, выходной `lm_head` в FP32).

## Готовая ONNX-модель

Веса модели опубликованы на Hugging Face:
👉 **[`davnozdubv/sage-ru-dictation-onnx`](https://huggingface.co/davnozdubv/sage-ru-dictation-onnx)**

| Файл | Размер | Назначение |
| :--- | :--- | :--- |
| `encoder_model.int8.onnx` | 43 МБ | Энкодер T5 (INT8) |
| `decoder_model.int8.onnx` | 148 МБ | Декодер T5 (INT8 + `lm_head` FP32) |
| `tokenizer.json` | 5 МБ | Fast BPE-токенизатор (совместим с Rust `tokenizers`) |

## Что умеет
* Расставляет знаки препинания (`. , ? ! — :`) и заглавные буквы.
* Различает **прямые вопросы (`?`)** и **косвенные вопросы / просьбы (`.`)** (10/10 на контрольном тесте).
* Исправляет опечатки, фонетические ошибки распознавания речи, повторы слов и неправильные русские окончания (падежи, род, число, глаголы).
* Переводит кириллическую транслитерацию IT/AI-брендов в правильный английский вид (`дипсик 4 1 флэш` → `DeepSeek 4.1 Flash`, `гитхаб` → `GitHub`, `гугл колаб` → `Google Colab`, `оннкс рантайм` → `ONNX Runtime`, `пайторч` → `PyTorch`, `апи` → `API` и др.).
* Не портит уже корректно оформленный текст.

## Производительность (Apple M2 Pro, CPU / ONNX Runtime)
* **Загрузка модели:** ~350 мс
* **Средняя задержка на фразу:** **~96 мс** (47–149 мс)

## Структура репозитория
* `build_dataset.py` — нарезка `.epub` на короткие русские отрывки (1–3 предложения) и генерация искажений в стиле GigaAM.
* `generate_dictation_synth.py` — генератор современных пар диктовки (опечатки, окончания, списки, косвенные/прямые вопросы, IT-бренды, самоисправления).
* `colab_benchmark_and_train.py` — бенчмарк (CER, WER, Punctuation F1, Exact Match) и обучение на GPU.
* `run_stage3_colab.py` — единый пайплайн совместного дообучения и экспорта в ONNX (INT8 + `lm_head` FP32).
* `test_local_onnx.py` — локальный тест и замер скорости ONNX-модели.

## Быстрый локальный тест

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python onnxruntime tokenizers numpy
.venv/bin/python test_local_onnx.py
```

## Лицензия
Базовая модель `ai-forever/sage-fredt5-distilled-95m` распространяется под лицензией MIT. Скрипты в этом репозитории также распространяются под лицензией MIT.
