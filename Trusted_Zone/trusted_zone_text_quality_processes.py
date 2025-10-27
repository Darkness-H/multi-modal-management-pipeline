# Importing useful dependencies
import base64
import io
import logging
import re
import time

import unicodedata

import boto3
import chardet
import pandas as pd
import tiktoken
from langdetect import detect_langs, detect
from deep_translator import GoogleTranslator
# set tokenizer with openAI standard token
from typing import List, Dict
from ftfy import fix_text
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Constant and function will support quality check
#Control characters
CTRL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
BIDI_RE = re.compile(r'[\u202A-\u202E\u2066-\u2069]')

# check api keys, and personal information
PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
}
SECRET_PATTERNS = {
    "api_key": r"\b(sk-|AKIA|AIza)[A-Za-z0-9-_]{10,}\b"
}

def language_check(text):
    """
    This function will detect languages present in text

    text        : str       - the text to be checked
    """
    try:
        langs = detect_langs(text)
        language_probs = {str(l.lang): l.prob for l in langs}
        mixed_language = len(language_probs) > 1 and max(language_probs.values()) < 0.95
    except Exception:
        language_probs, mixed_language = {}, False
    return language_probs, mixed_language


def get_text(client, bucket, key):
    """
     A simple function to get text from an S3 object

    client     : obj         - S3-compatible client (e.g., boto3.client("s3")).
    bucket     : str         - Target S3/MinIO bucket.
    key        : str         - path of the file
    """
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    text = body.decode("utf-8")
    return text

# ---- Quality checks ----
def check_text_quality(body: bytes, key: str):
    """
    Return a dictionary of essential text quality metrics for a single file.

    Part of the Trusted Zone data validation layer, this function performs
    structural, encoding, and content-level checks to detect corrupted or
    low-quality text before it moves further in the data pipeline.
    It is model-agnostic, focusing on intrinsic data issues such as
    encoding errors, unreadable characters, abnormal structure, and
    mixed-language content.

    body        : bytes       - raw file content as bytes
    key        : str         - path of the file
    """
    if not body:
        return {"key": key, "empty": True}

    # detect encoding and decode safely
    guess = chardet.detect(body)
    enc = guess.get("encoding") or "utf-8"
    encoding_confidence = guess.get("confidence", 0.0)
    has_bom = body.startswith((b'\xef\xbb\xbf', b'\xff\xfe', b'\xfe\xff'))

    text = body.decode(enc, errors="replace")
    # check control character
    control_char_ratio = len(CTRL_RE.findall(text)) / max(1, len(text))
    has_null_bytes = '\x00' in text
    bidi_ctrl_count = len(BIDI_RE.findall(text))
    # check line endings
    line_endings = {
        "LF": text.count("\n"),
        "CRLF": text.count("\r\n"),
        "CR": text.count("\r") - text.count("\r\n"),
    }
    mixed_line_endings = sum(v > 0 for v in line_endings.values()) > 1
    # check printable ratio (avoid binary garbage)
    printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / max(1, len(text))

    # check line stats
    lines = text.splitlines()
    line_lengths = [len(l) for l in lines if l.strip()]
    avg_line_len = np.mean(line_lengths) if line_lengths else 0
    p95_line_len = np.percentile(line_lengths, 95) if line_lengths else 0
    max_line_len = max(line_lengths, default=0)
    # --- Paragraph token stats ---
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    paragraph_tokens = [len(p) for p in paragraphs] if paragraphs else []

    avg_tokens_per_paragraph = (
        sum(paragraph_tokens) / len(paragraph_tokens) if paragraph_tokens else 0
    )
    max_tokens_paragraph = max(paragraph_tokens) if paragraph_tokens else 0
    min_tokens_paragraph = min(paragraph_tokens) if paragraph_tokens else 0
    # basic stats
    sentences = re.split(r"[.!?。！？]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # unicode check
    normalized_text = unicodedata.normalize("NFKC", text)
    diff = sum(1 for a, b in zip(text, normalized_text) if a != b)
    norm_changed_ratio = diff / max(1, len(text))

    # language check
    language_probs, mixed_language = language_check(text)

    # legislation
    pii_hits = {k: len(re.findall(v, text)) for k, v in PII_PATTERNS.items()}
    secret_hits = {k: len(re.findall(v, text)) for k, v in SECRET_PATTERNS.items()}
    stats = {
        "key": key,

        # === 1. Encoding Layer ===
        "size_bytes": len(body),
        "encoding": enc,
        "encoding_confidence": encoding_confidence,
        "has_bom": has_bom,

        # === 2. Control & Binary Layer ===
        "control_char_ratio": control_char_ratio,
        "has_null_bytes": has_null_bytes,
        "bidi_ctrl_count": bidi_ctrl_count,
        "printable_ratio": printable_ratio,

        # === 3. Structure Layer ===
        "lines": len(lines),
        "mixed_line_endings": mixed_line_endings,
        "avg_line_len": avg_line_len,
        "p95_line_len": p95_line_len,
        "max_line_len": max_line_len,

        # === 4. Readability / Paragraph Layer ===
        "paragraphs": len(paragraphs),
        "avg_tokens_per_paragraph": round(avg_tokens_per_paragraph, 2),
        "paragraph_token_list": paragraph_tokens,
        "max_tokens_paragraph": max_tokens_paragraph,
        "min_tokens_paragraph": min_tokens_paragraph,
        "sentence_count": sentence_count,

        # === 5. Unicode Layer ===
        "norm_changed_ratio": norm_changed_ratio,

        # === 6. Language Layer ===
        "language_probs": language_probs,
        "mixed_language": mixed_language,

        # === 7. Content Quality Layer ===
        "too_short": len(text.strip()) < 20,
        "empty": not bool(text.strip()),

        # === 8. Compliance Layer ===
        "pii_hits": pii_hits,
        "secret_hits": secret_hits,
    }
    return stats


def extract_datas(client, bucket, prefix=""):
    """
    List objects under `prefix`, filter to text, and return basic text metadata.

    client     : obj         - S3-compatible client (e.g., boto3.client("s3")).
    bucket     : str         - Target S3/MinIO bucket.
    prefix     : str         - Optional key prefix (acts like a folder path).
    """
    t0=time.perf_counter()
    summary = {
        "scanned": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
    }
    results = []
    etags = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            summary["scanned"] += 1
            duplicate = False
            key = obj["Key"]

            size = obj["Size"]
            etag = obj["ETag"].strip('"')
            if size == 0 and key.endswith("/"):  # skip the folder itself
                summary["skipped"] += 1
                continue

            sig = (etag, size)
            if sig in etags:
                duplicate = True
            else:
                etags.add(sig)
            # Download the text
            try:
                resp = client.get_object(Bucket=bucket, Key=key)
                body = resp["Body"].read()
            except Exception as e:
                summary["failed"] += 1
                logger.exception("Failed to process %s: %s", key, e)
                continue
            stats = check_text_quality(body, key)
            stats["duplicated"] = duplicate
            results.append(stats)
            summary["succeeded"] += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Texts data extraction completed in %.2fs — scanned=%d, succeeded=%d, skipped=%d, failed=%d",
        elapsed, summary["scanned"], summary["succeeded"],
        summary["skipped"], summary["failed"]
    )

    return results


def generate_quality_report(df: pd.DataFrame, name):
    """
    Generate and display an interactive data quality report for Trusted Zone text data.
    """
    # ---- Validate input ----
    if df.empty:
        print(" The DataFrame is empty. Nothing to analyze.")
        return None

    # ----  Compute key issue counts ----
    df["pii_detected"] = df["pii_hits"].apply(lambda x: sum(x.values()) > 0)
    summary = {
        "total_files": len(df),
        "empty_or_short": (df["empty"] | df["too_short"]).sum(),
        "low_printable_ratio": df[df["printable_ratio"] < 0.9].shape[0],
        "high_norm_ratio": (df["norm_changed_ratio"] > 0.3).sum(),
        "mixed_language": df["mixed_language"].sum(),
        "pii_detected": df["pii_detected"].sum(),
        "control_chars": (df["control_char_ratio"] > 0.01).sum(),
        "null_bytes": df["has_null_bytes"].sum(),
        "bidi_ctrl": (df["bidi_ctrl_count"] > 0).sum(),
    }

    summary_df_html = pd.DataFrame(summary.items(), columns=["Metric", "Count"]).to_html(border = 0)
    desc_html = df.describe(include="all").to_html(border=0)
    # ----  Distributions ----
    plots = {}
    def fig_to_base64(figure):
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight")
        plt.close(figure)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["printable_ratio"], bins=20, ax=ax)
    ax.set_title("Printable Ratio Distribution")
    ax.set_xlabel("Printable Ratio")
    ax.set_ylabel("Files")
    plots["printable_ratio_dist"] = fig_to_base64(fig)

    plt.subplot(1, 3, 2)
    plt.hist(df["norm_changed_ratio"], bins=20, color='orange')
    plt.title("Unicode Normalization Change Ratio")
    plt.xlabel("norm_changed_ratio")
    plt.ylabel("Files")

    plt.subplot(1, 3, 3)
    sns.boxplot(x=df["avg_line_len"], color='lightblue')
    plt.title("Average Line Length (Boxplot)")
    plt.tight_layout()
    plt.show()

    # ----  Language probability distribution ----
    all_langs = {}
    for d in df["language_probs"]:
        for lang, prob in d.items():
            all_langs[lang] = all_langs.get(lang, 0) + prob
    if all_langs:
        plt.figure(figsize=(6, 4))
        pd.Series(all_langs).sort_values(ascending=True).plot.barh(
            title="Detected Language Probability Sum", color='teal'
        )
        plt.xlabel("Summed Probability Across Files")
        plt.show()

    # ----  PII frequency summary ----
    pii_counts = pd.DataFrame(df["pii_hits"].tolist()).sum().sort_values(ascending=False)
    if pii_counts.sum() > 0:
        plt.figure(figsize=(5, 3))
        pii_counts.plot.bar(title="Detected PII Counts", color='salmon')
        plt.ylabel("Occurrences")
        plt.show()

    # ----  Correlation heatmap between quality issues ----
    issue_flags = pd.DataFrame({
        "low_printable_ratio": df["printable_ratio"] < 0.9,
        "high_norm_ratio": df["norm_changed_ratio"] > 0.3,
        "mixed_language": df["mixed_language"],
        "pii": df["pii_detected"],
        "null_bytes": df["has_null_bytes"],
        "control_chars": df["control_char_ratio"] > 0.01
    }).fillna(0).astype(int)

    corr = issue_flags.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Quality Issues")
    plt.show()

    # ----  Print summary overview ----
    print("\n Data Quality Report generated successfully.\n")
    print(f"Total files analyzed: {len(df)}")
    low_printable = (df["printable_ratio"].fillna(0) < 0.9)
    high_norm = (df["norm_changed_ratio"].fillna(0) > 0.3)
    pii_detected = df["pii_detected"].fillna(False).astype(bool)

    potential_issues = (low_printable | high_norm | pii_detected).sum()

    print(f"Files with potential issues: {int(potential_issues)}")


WS_RE = re.compile(r"[ \t\u00A0\u2000-\u200B\u3000]+")


def basic_clean(text: str) -> str:
    # Normalize & strip BOM
    text = unicodedata.normalize("NFKC", text).replace("\ufeff", "")

    # Remove control chars (keep whitespace)
    text = CTRL_RE.sub("", text)

    # Normalize newlines first (helps regex around lines)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # PII redaction (email/phone)
    for pattern in PII_PATTERNS.values():
        text = re.sub(pattern, "", text)

    # Line-wise whitespace normalization
    #  - collapse runs of spaces/tabs/various unicode spaces into a single space
    #  - strip ends
    text = "\n".join(WS_RE.sub(" ", ln).strip() for ln in text.split("\n"))

    # Fix artifacts after redaction: dangling @, +, (), multiple spaces, stray punctuation
    text = re.sub(r"\s{2,}", " ", text)  # collapse multi-spaces
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)  # remove space before punctuation
    text = re.sub(r"([(@+])\s*(?=[)@+.,;:!?])", r"", text)  # clear leftover symbols clusters
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines

    return text


HIGH_UNICODE_GARBAGE_RE = re.compile(r"[^\x00-\x7F]+")
COMBINING_MARKS_RE = re.compile(r'[\u0300-\u036F]+')  # combining marks

try:
    import emoji

    EMOJI_RE = emoji.get_emoji_regexp()
except Exception:
    EMOJI_RE = re.compile(r'[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+')

REPLACE_TO_ASCII = {
    "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00A0": " ",
}


def _map_preferred_ascii(s: str) -> str:
    if not s:
        return s
    return s.translate(str.maketrans(REPLACE_TO_ASCII))


def deep_clean(text: str, ascii_only: bool = True, remove_emoji: bool = True, drop_combining: bool = True) -> str:
    """
    Aggressive cleanup. Use AFTER translation to English.
    """
    #  Prefer ASCII-friendly replacements before filtering
    text = _map_preferred_ascii(text)
    # 2) Remove emoji (optional)
    if remove_emoji:
        text = EMOJI_RE.sub(" ", text)
    #  Remove combining marks (optional)
    if drop_combining:
        text = COMBINING_MARKS_RE.sub("", text)
    #  Enforce ASCII-only (optional & aggressive)
    if ascii_only:
        text = HIGH_UNICODE_GARBAGE_RE.sub(" ", text)
    #  Final tidy
    text = unicodedata.normalize("NFKC", text)  # keep stability
    text = CTRL_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# %%
LANG_MAP = {
    "zh-cn": "zh-CN",
    "zh_cn": "zh-CN",
    "zh-CN": "zh-CN",
    "zh": "zh-CN",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "ja": "ja",
    "ko": "ko"
}


def to_english(text: str, lang, sentence=False) -> str:
    try:
        if not sentence:
            src_lang = LANG_MAP.get(lang.lower(), lang)
            result = GoogleTranslator(source=src_lang, target='en').translate(text)
        else:
            result = GoogleTranslator(source="auto", target='en').translate(text)
        return result, True
    except Exception as e:
        print(f"Translation failed ({e}), keeping original text.")
        return text, False


_SENT_SPLIT_RE = re.compile(
    r'(?:\s*\n+\s*)|'  # paragraph/newline breaks
    r'(?<=[。！？!?])\s+'  # CJK sentence enders + whitespace
    r'|(?<=[\.\?\!])\s+'  # English . ? ! + whitespace
)


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r'\r\n?', '\n', text).strip()
    parts = []
    for para in filter(None, text.split('\n')):
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue
        parts.extend([s.strip() for s in re.split(_SENT_SPLIT_RE, para) if s.strip()])
    return parts


def chunk_text_with_token_budget(
        text: str,
        tokenizer,
        max_tokens: int = 512,
        stride: int = 64,
        reserve_special: int = 2,  # reserve tokens for [CLS]/[SEP] or similar
) -> List[Dict]:
    # The actual available token budget after reserving special tokens
    budget = max_tokens - reserve_special
    assert budget > 0, "max_tokens is too small; cannot reserve special tokens"

    # Split text into sentences
    sents = split_into_sentences(text)

    chunks: List[Dict] = []
    curr_sents: List[str] = []
    curr_tokens = 0

    # Helper: count token length for a given text (excluding special tokens)
    def token_len(txt: str) -> int:
        return len(tokenizer.encode(txt))

    # Precompute token lengths of all sentences for efficiency
    sent_lens = [token_len(s) for s in sents]

    i = 0
    while i < len(sents):
        s = sents[i]
        s_len = sent_lens[i]

        # Case A: single sentence exceeds budget → split within sentence (sliding window)
        if s_len > budget:
            # If current chunk buffer is not empty, finalize it first
            if curr_sents:
                chunk_text = " ".join(curr_sents).strip()
                chunks.append({"text": chunk_text, "n_tokens": token_len(chunk_text), "input_ids": None})
                curr_sents, curr_tokens = [], 0

            # Encode the long sentence into token IDs
            input_ids = tokenizer.encode(s)
            start = 0
            while start < len(input_ids):
                # Take a slice of up to `budget` tokens
                end = min(start + budget, len(input_ids))
                piece_ids = input_ids[start:end]
                # Decode back to text for saving
                piece_text = tokenizer.decode(piece_ids).strip()
                chunks.append({"text": piece_text, "n_tokens": len(piece_ids), "input_ids": None})

                # If reached the end, stop
                if end == len(input_ids):
                    break
                # Move window forward with overlap (`stride`)
                start = max(end - stride, start + 1)
            i += 1
            continue

        # Case B: greedy packing — keep adding sentences to current chunk
        if curr_tokens + s_len <= budget:
            curr_sents.append(s)
            curr_tokens += s_len
            i += 1
        else:
            # Finalize the current chunk when adding the next sentence would exceed the budget
            chunk_text = " ".join(curr_sents).strip()
            chunks.append({"text": chunk_text, "n_tokens": token_len(chunk_text), "input_ids": None})
            curr_sents, curr_tokens = [], 0

    # Finalize any remaining sentences
    if curr_sents:
        chunk_text = " ".join(curr_sents).strip()
        chunks.append({"text": chunk_text, "n_tokens": token_len(chunk_text), "input_ids": None})

    return chunks


def clean_text(client,bucket,prefix , max_tokens=512):
    bucket = "trusted-zone"
    data = extract_datas(client, bucket, prefix,"before_clean")
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if df.empty:
        raise ValueError("No text data to report.")
    generate_quality_report(df)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for row in df.itertuples(index=False):

        # check text need remove or clean
        if row.empty or row.too_short or row.printable_ratio < 0.9 or row.duplicated:
            client.delete_object(Bucket=bucket, Key=row.key)
            continue
        # get text
        text = get_text("trusted-zone", row.key)
        # basic clean and get text
        text = basic_clean(text)

        # Language gate: decide whether this document likely needs translation to English
        # Uses overall English probability from row.language_probs; if below 0.85, enable translation.
        should_translate = False
        prob = row.language_probs.get("en", 0)
        if prob < 0.85:
            should_translate = True

        # Split the full text by blank lines into paragraphs (preserve only non-empty ones)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i in range(0, len(row.paragraph_token_list)):
            # Translate before token-based chunking:
            # doing translation first avoids re-breaking chunks after token counts change due to translation.
            if should_translate:
                paragraph = paragraphs[i]
                # Detect the source language at the paragraph level (more precise than doc-level for mixed content)
                source = detect(paragraph)

                # If this paragraph is not English, attempt paragraph-level translation first
                if (source != "en"):
                    # Primary attempt: translate the whole paragraph
                    result, sucess = to_english(paragraph, source)

                    # Fallback: if paragraph translation fails, translate sentence-by-sentence and then aggregate back
                    if not sucess:
                        senteces = split_into_sentences(paragraph)
                        translated_sentences = []
                        for sent in senteces:
                            source = detect(sent)
                            translated = to_english(sent, source, sentence=True)
                            translated_sentences.append(translated.strip())

                        # Reassemble the paragraph from translated sentences (simple space join)
                        result = ".".join(s for s in translated_sentences if s)

                    # Replace the original paragraph with the translated result
                    paragraphs[i] = result

            # Token-length control:
            # If the original paragraph's token count exceeded `max_tokens`, chunk the (possibly translated) paragraph.
            if (row.paragraph_token_list[i] > max_tokens):
                chunks = chunk_text_with_token_budget(paragraphs[i], tokenizer, max_tokens)
                # Join chunks with double newlines to preserve a visual boundary between slices
                full_text = "\n\n".join(chunk["text"].strip() for chunk in chunks if chunk["text"].strip())
                paragraphs[i] = full_text

        # Reassemble the full text by joining paragraphs with blank lines
        text = "\n\n".join(paragraphs)

        # Post-translation validation:
        # Check language again to verify the text is predominantly English and not mixed-language.
        language_probs, mixed_languages = language_check(text)

        # Last-resort cleanup:
        # If still mixed-language or English probability is too low, try textual fixes and deep cleaning,
        # then re-check. If still not acceptable, delete the object.
        # (Typical causes: rare code fragments, heavy non-ASCII characters, odd punctuation.)
        if (mixed_languages or language_probs.get("en", 0) < 0.85):
            fixed = fix_text(text)
            text = deep_clean(fixed)
            language_probs, mixed_languages = language_check(text)
            # Final rejection if it still fails the language criteria
            if (mixed_languages or language_probs.get("en", 0) < 0.85):
                client.delete_object(Bucket=bucket, Key=row.key)
                continue
        # accept text
        client.put_object(
            Bucket=bucket,
            Key=row.key,  # Make sure the file key (path) is correct
            Body=text.encode('utf-8'),
            ContentType="text/plain"
        )
        print(f"Normalized: {row.key}")

        data = extract_datas(client, bucket, prefix)
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
        if df.empty:
            raise ValueError("No text data to report.")
        generate_quality_report(df,"after_clean")

