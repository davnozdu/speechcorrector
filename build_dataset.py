#!/usr/bin/env python3
"""
Извлечение коротких русских отрывков (1-3 предложения) из .epub книг
и генерация пар (испорченный вход в стиле GigaAM -> чистый оригинал)
для бенчмарка и дообучения ai-forever/sage-fredt5-distilled-95m.
"""

import argparse
import html
import json
import os
import random
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "br"}
    SKIP_TAGS = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__()
        self.paragraphs: List[str] = []
        self.current: List[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        elif tag in self.BLOCK_TAGS and self.skip_depth == 0:
            self._flush()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
        elif tag in self.BLOCK_TAGS and self.skip_depth == 0:
            self._flush()

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.current.append(data)

    def _flush(self):
        text = "".join(self.current).strip()
        if text:
            self.paragraphs.append(text)
        self.current = []

    def get_paragraphs(self) -> List[str]:
        self._flush()
        return self.paragraphs


CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
LETTER_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁÀ-ž]")
FOOTNOTE_RE = re.compile(r"\[\d+\]|\(\d+\)")
MULTI_SPACE_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁ«\"(])")

# Частые аббревиатуры, после которых точка не является концом предложения
ABBREV_RE = re.compile(r"\b(г|ул|им|стр|руб|коп|т\.д|т\.п|т\.е|и др|см|ср|пр|др|гг)\.$", re.IGNORECASE)


def clean_paragraph(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\xa0", " ")
    text = FOOTNOTE_RE.sub("", text)
    # Нормализуем тире и кавычки
    text = text.replace("–", "—").replace("―", "—")
    text = text.replace("«", "«").replace("»", "»")
    text = text.replace("...", "…")
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    # Убираем ведущее книжное тире диалога, чтобы текст был ближе к диктовке
    text = re.sub(r"^[—\-]\s+", "", text)
    return text


def is_clean_russian(text: str) -> bool:
    letters = LETTER_RE.findall(text)
    if len(letters) < 15:
        return False
    cyr = CYRILLIC_RE.findall(text)
    if len(cyr) / len(letters) < 0.90:
        return False
    # Отсекаем строки с избытком спецсимволов или цифр (оглавления, библиография)
    if any(ch in text for ch in ("*", "_", "/", "\\", "=", "{", "}", "|", "<", ">", "@", "#")):
        return False
    digits = sum(c.isdigit() for c in text)
    if digits / max(len(text), 1) > 0.15:
        return False
    # Отсекаем строки капсом (заголовки глав)
    upper = sum(c.isupper() for c in cyr)
    if upper / len(cyr) > 0.35:
        return False
    return True


def split_sentences(paragraph: str) -> List[str]:
    parts = SENT_SPLIT_RE.split(paragraph)
    merged: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and ABBREV_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def extract_chunks_from_epub(epub_path: Path, rng: random.Random) -> List[str]:
    chunks: List[str] = []
    try:
        zf = zipfile.ZipFile(epub_path, "r")
    except Exception:
        return chunks

    with zf:
        names = [
            n for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm"))
            and "toc" not in n.lower()
            and "nav" not in n.lower()
        ]
        for name in sorted(names):
            try:
                raw = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            parser = TextExtractor()
            try:
                parser.feed(raw)
            except Exception:
                continue
            for p in parser.get_paragraphs():
                p = clean_paragraph(p)
                if not is_clean_russian(p):
                    continue
                sents = [s for s in split_sentences(p) if is_clean_russian(s)]
                i = 0
                while i < len(sents):
                    # Собираем 1-3 предложения, как при диктовке
                    k = rng.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0]
                    chunk = " ".join(sents[i : i + k]).strip()
                    words = chunk.split()
                    if 4 <= len(words) <= 40 and 20 <= len(chunk) <= 280:
                        # Пропускаем обрывки без нормального конца предложения
                        if chunk[-1] in ".!?…»\"":
                            chunks.append(chunk)
                    i += k
    return chunks


# --- Генерация реалистичных искажений (GigaAM + ASR-ошибки) ---

PUNCT_STRIP_RE = re.compile(r"[.,?!:;…—«»\"()\[\]]+")

PHONETIC_SUBS = [
    ("ться", "тся"),
    ("тся", "ться"),
    ("нн", "н"),
    ("ё", "е"),
    ("жи", "жы"),
    ("ши", "шы"),
    ("о", "а"),
    ("а", "о"),
    ("е", "и"),
    ("и", "е"),
    ("т", "д"),
    ("д", "т"),
    ("п", "б"),
    ("б", "п"),
    ("к", "г"),
    ("г", "к"),
    ("с", "з"),
    ("з", "с"),
    ("ш", "щ"),
    ("щ", "ш"),
]

COMMON_MERGES = [
    ("не ", "не"),
    ("в ", "в"),
    ("на ", "на"),
    ("по ", "по"),
    ("потому что", "потомучто"),
    ("так как", "таккак"),
    ("то есть", "тоесть"),
    ("как будто", "какбудто"),
    ("может быть", "можетбыть"),
    ("всё равно", "всеравно"),
    ("все равно", "всеравно"),
]


def strip_all_punct(text: str, rng: random.Random) -> str:
    s = PUNCT_STRIP_RE.sub(" ", text)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    s = s.lower()
    if rng.random() < 0.65:
        s = s.replace("ё", "е")
    return s


def partial_gigaam_corrupt(text: str, rng: random.Random) -> str:
    """Имитирует частичную/кривую пунктуацию GigaAM:
    пропуск запятых, случайные точки вместо запятых, сбитый регистр.
    """
    # Убираем кавычки, двоеточия, точки с запятой, тире
    s = re.sub(r"[«»\"():;—]", " ", text)
    s = s.replace("…", "." if rng.random() < 0.5 else "")

    # Пропускаем большинство запятых
    def corrupt_comma(m):
        r = rng.random()
        if r < 0.70:
            return ""
        elif r < 0.85:
            return "."
        return ","

    s = re.sub(r",", corrupt_comma, s)

    # Иногда убираем точку/вопрос/восклицание или меняем ? на .
    def corrupt_end(m):
        ch = m.group(0)
        r = rng.random()
        if ch in ("?", "!") and r < 0.55:
            return "." if rng.random() < 0.5 else ""
        if r < 0.45:
            return ""
        return ch

    s = re.sub(r"[.?!]", corrupt_end, s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()

    # Сбиваем регистр
    words = s.split()
    out_words = []
    for idx, w in enumerate(words):
        if rng.random() < 0.65:
            w = w.lower()
        elif idx > 0 and rng.random() < 0.06 and len(w) > 2:
            w = w[0].upper() + w[1:].lower()
        out_words.append(w)
    s = " ".join(out_words)
    if rng.random() < 0.6:
        s = s.replace("ё", "е")
    return s


def inject_asr_word_errors(text: str, rng: random.Random, n_errors: int = 1) -> str:
    s = text
    # Иногда склеиваем предлог/союз
    if rng.random() < 0.35:
        candidates = [(a, b) for a, b in COMMON_MERGES if a in s]
        if candidates:
            a, b = rng.choice(candidates)
            s = s.replace(a, b, 1)

    words = s.split()
    if len(words) < 3:
        return s

    for _ in range(n_errors):
        idxs = [i for i, w in enumerate(words) if len(w) >= 4 and CYRILLIC_RE.search(w)]
        if not idxs:
            break
        idx = rng.choice(idxs)
        w = words[idx]
        op = rng.random()
        if op < 0.50:
            # Фонетическая замена внутри слова
            applicable = [(a, b) for a, b in PHONETIC_SUBS if a in w[1:]]
            if applicable:
                a, b = rng.choice(applicable)
                pos = w.find(a, 1)
                if pos != -1:
                    w = w[:pos] + b + w[pos + len(a) :]
        elif op < 0.75 and len(w) >= 5:
            # Перестановка двух соседних букв (опечатка)
            pos = rng.randint(1, len(w) - 3)
            w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
        elif len(w) >= 5:
            # Пропуск одной буквы
            pos = rng.randint(1, len(w) - 2)
            w = w[:pos] + w[pos + 1 :]
        words[idx] = w
    return " ".join(words)


def corrupt_chunk(target: str, rng: random.Random) -> Tuple[str, str]:
    r = rng.random()
    if r < 0.45:
        return strip_all_punct(target, rng), "strip_all"
    elif r < 0.75:
        return partial_gigaam_corrupt(target, rng), "gigaam_partial"
    elif r < 0.92:
        base = strip_all_punct(target, rng) if rng.random() < 0.5 else partial_gigaam_corrupt(target, rng)
        n_err = 1 if len(target.split()) < 15 else rng.choice([1, 2])
        return inject_asr_word_errors(base, rng, n_errors=n_err), "asr_errors"
    else:
        # Чистый текст (чтобы модель не портила уже правильные фразы)
        return target, "identity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-train", type=int, default=45000)
    ap.add_argument("--val-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    epubs = sorted(args.epub_dir.glob("*.epub"))
    print(f"Found {len(epubs)} epub files in {args.epub_dir}")

    seen = set()
    all_chunks: List[Tuple[str, str]] = []  # (book_name, chunk)
    book_counts = {}

    for epub in epubs:
        chunks = extract_chunks_from_epub(epub, rng)
        added = 0
        for ch in chunks:
            key = ch.lower()
            if key not in seen:
                seen.add(key)
                all_chunks.append((epub.name, ch))
                added += 1
        if added > 0:
            book_counts[epub.name] = added

    print(f"Extracted {len(all_chunks)} unique Russian chunks from {len(book_counts)} books")
    for b, cnt in sorted(book_counts.items(), key=lambda x: -x[1])[:12]:
        print(f"  {cnt:6d}  {b}")

    rng.shuffle(all_chunks)

    val_items = all_chunks[: args.val_size]
    train_items = all_chunks[args.val_size : args.val_size + args.max_train]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_split(path: Path, items: List[Tuple[str, str]]):
        mode_stats = {}
        with path.open("w", encoding="utf-8") as f:
            for book, target in items:
                source, mode = corrupt_chunk(target, rng)
                mode_stats[mode] = mode_stats.get(mode, 0) + 1
                rec = {"source": source, "target": target, "mode": mode, "book": book}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items)} rows to {path} | modes: {mode_stats}")

    write_split(args.out_dir / "train.jsonl", train_items)
    write_split(args.out_dir / "val.jsonl", val_items)

    # Показываем 5 примеров для визуальной проверки
    print("\n--- Sample val pairs ---")
    for book, target in val_items[:5]:
        src, mode = corrupt_chunk(target, rng)
        print(f"[{mode}]")
        print(f"  IN : {src}")
        print(f"  OUT: {target}")


if __name__ == "__main__":
    main()
