# Importing useful dependencies
import logging
import math
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import psutil
import torch
import open_clip
import numpy as np


from src.exploitation_zone.utils_exploitation.embeddings import embed_image,embed_text
from src.exploitation_zone.utils_exploitation.getter import get_text, get_image
from src.utils.file_utils import fmt_bytes

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load "ViT-B-16" model for images and texts
model_it, _, preprocess_it = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tokenizer_it = open_clip.get_tokenizer("ViT-B-16") # Tokenizer for texts
model_it.to(device)


def print_top_k_files(res,client , k=3):
    """
    Display and collect top-K search results (texts & images) from a Chroma query result.

    res     : dict      - Chroma query result with keys like "documents" and "distances".
    client  : obj       - S3-compatible client used by get_text() / get_image().
    k       : int       - Maximum number of texts and images to display and collect.
    """
    # Print results with type
    n_text = 0
    n_image = 0
    i = 0
    images = []
    results =""
    results_img = ""
    for _, doc in enumerate(res["documents"][0]):
        if (doc.split(".")[-1] == "txt" and n_text < k):
            text = ""
            text += (f"{i + 1}. Distance: {res['distances'][0][i]:.4f}")
            text += "\n"
            text += (f"Content: {doc.replace("trusted-zone/", "", 1)}")
            text += "\n"
            text +=(get_text(client,"trusted-zone", doc.replace("trusted-zone/", "", 1)))
            text += "\n"
            text += ("-" * 40)
            text += "\n"

            print(text)
            results += text

            i += 1
            n_text += 1

        elif (doc.split(".")[-1] == "png" and n_image < k):
            text = ""
            text += (f"{i + 1}. Distance: {res['distances'][0][i]:.4f}")
            text += "\n"
            text += (f"Content: {doc.replace("trusted-zone/", "", 1)}")
            text += "\n"
            results_img += text
            print(text)
            img = get_image(client,"trusted-zone", doc.replace("trusted-zone/", "", 1))  # PIL.Image
            images.append((img,doc))
            print("-" * 40)
            i += 1
            n_image += 1

        # Stop early if both top-k limits are reached
        if n_text >= k and n_image >= k:
                break
    if images:
        n = len(images)
        cols = min(3, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 1, 3.5 * rows))
        # Normalize axes to 2D array
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        idx = 0
        for img, title in images:
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=9)
            idx += 1

        # Hide unused axes
        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            axes[r][c].axis("off")

        fig.suptitle(f"Top-{n} Images (each ≤ {k})", fontsize=12)
        fig.tight_layout()
        plt.show(block=True)

    return results, results_img


def find_similar_files(s3_client, collection, query_emb: np.ndarray,k = 3):
    # Chroma expects list-of-lists for query_embeddings
    query_vector = query_emb.tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2000,
        include=["documents", "distances"]
    )

    res1,res2 = print_top_k_files(results,s3_client,k)

    return res1,res2

def find_k_similars_by_file(s3_client, collection, doc, doc_type,k = 3):
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    tracemalloc.start()
    logger.info("Start multi-modal similarity search")
    query_emb = None
    if doc_type == "image":
        query_emb = embed_image(preprocess_it, model_it,doc,device)

    elif doc_type == "text":
        query_emb = embed_text(tokenizer_it, model_it,doc,device)

    if query_emb is not None:
        res1,res2 = find_similar_files(s3_client,collection,query_emb,k)
        return res1,res2
    # time
    elapsed = time.perf_counter() - t0
    # memory
    current_rss = proc.memory_info().rss
    rss_diff = current_rss - rss_before
    # python heap peak
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(
        "METRICS | elapsed=%0.2f s | rss_before=%s | rss_after=%s | rss_diff=%s | py_heap_peak=%s",
        elapsed,
        fmt_bytes(rss_before),
        fmt_bytes(current_rss),
        fmt_bytes(rss_diff),
        fmt_bytes(peak),
    )
    return None, None
