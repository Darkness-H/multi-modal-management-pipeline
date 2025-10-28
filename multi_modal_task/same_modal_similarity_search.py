# Importing useful dependencies
import logging
import math
import os
import time
import tracemalloc

import cv2
import open_clip
import psutil
import torch

import numpy as np

from matplotlib import pyplot as plt
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from exploitation_zone.utils_exploitation.getter import get_text, get_image,get_video
from exploitation_zone.utils_exploitation.embeddings import embed_text,embed_image,embed_video
from utils.file_utils import fmt_bytes

logger = logging.getLogger(__name__)

def play_frames_opencv(frames, win_name="Frames"):
    """
    Play a folder of frame images (png/jpg/...) as a video using an OpenCV window.

    Controls:
      - Space: pause/resume
      - q / Esc: quit
      - +/- : speed down/up (change delay)
    """
    delay_ms = max(1, int(1000.0 / max(1e-6, 1)))  # ms per frame fps = 1 because we normalized videos in our dataset
    paused = False
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # resizable window

    for frame in frames:
        # read one frame (bytes -> ndarray)

        if frame is None:
            continue

        cv2.imshow(win_name, frame)

        # keyboard control
        t = 1 if paused else delay_ms
        key = cv2.waitKey(t) & 0xFF
        if key in (ord('q'), 27):   # q or Esc
            break
        elif key == 32:             # Space
            paused = not paused
        elif key == ord('+'):
            delay_ms = max(1, delay_ms - 5)
        elif key == ord('-'):
            delay_ms = min(100, delay_ms + 5)

    cv2.destroyAllWindows()


def find_similar_files(s3_client, collection, query_emb: np.ndarray, top_k: int = 5):
    # Chroma expects list-of-lists for query_embeddings
    query_vector = query_emb.tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"]
    )

    # Extract first query results
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    res = []
    print(f"Top {top_k} similar {collection.name}:")
    for rank, (doc_id, doc, dist) in enumerate(zip(ids, docs, dists), start=1):
        print(f"{rank}. id={doc_id}, distance={dist:.4f}")
        print(f"   document: {doc}")
        temp_path = doc.split("/")[-1]
        if (collection.name == "images"):
            image = (get_image(s3_client,"trusted-zone", f"{collection.name}/{temp_path}"))
            res.append((image,doc))
        elif (collection.name == "texts"):
            print("\n" + get_text(s3_client,"trusted-zone", f"{collection.name}/{temp_path}") + "\n")
        elif (collection.name == "videos"):
            frames = get_video(s3_client,"trusted-zone", f"{collection.name}/{temp_path}")
            play_frames_opencv(frames)

    if res:
        n = len(res)
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
        for img, title in res:
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

        fig.suptitle(f"Top-{n} Images (each ≤ {5})", fontsize=12)
        fig.tight_layout()
        plt.show(block=True)
    return results


# Just in case our device has gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
model_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai",quick_gelu=True)
tokenizer = open_clip.get_tokenizer("ViT-B-16") # Tokenizer for texts
model_text.to(device)

def get_similar_text(s3_client, colleccion,text):
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    tracemalloc.start()
    logger.info("Start similarity search")
    query_embed = embed_text(tokenizer, model_text, text, device)
    res_text = find_similar_files(s3_client, colleccion, query_embed, top_k=3)
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
    return res_text

# Load model + preprocessing used when creating the image embeddings
model_image, _, preprocess_image = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai",quick_gelu=True)
model_image.to(device)

def get_similar_image(s3_client, colleccion,image):
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    tracemalloc.start()
    logger.info("Start similarity search")
    query_embed = embed_image(preprocess_image, model_image, image, device)
    res_image = find_similar_files(s3_client, colleccion, query_embed, top_k=3)
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
    return res_image


preprocess_video = CLIPImageProcessor.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model_video = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model_video.to(device)

def get_similar_video(s3_client, colleccion,video_frames):
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    tracemalloc.start()
    logger.info("Start similarity search")
    query_embed = embed_video(preprocess_video, model_video, video_frames, device)
    res_video = find_similar_files(s3_client, colleccion, query_embed, top_k=3)
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
    return res_video
