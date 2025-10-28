# Importing useful dependencies
import math
import matplotlib.pyplot as plt
import torch
import open_clip
import numpy as np


from exploitation_zone.utils_exploitation.embeddings import embed_image,embed_text
from exploitation_zone.utils_exploitation.getter import get_text, get_image

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load "ViT-B-16" model for images and texts
model_it, _, preprocess_it = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tokenizer_it = open_clip.get_tokenizer("ViT-B-16") # Tokenizer for texts
model_it.to(device)


def print_top_k_files(res,client , k=5):
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
    titles = []
    for _, doc in enumerate(res["documents"][0]):
        if (doc.split(".")[-1] == "txt" and n_text < k):
            print(f"{i + 1}. Distance: {res['distances'][0][i]:.4f}")
            print("Content:", doc)
            print(get_text(client,"trusted-zone", doc.replace("trusted-zone/", "", 1)))
            print("-" * 40)
            i += 1
            n_text += 1

        elif (doc.split(".")[-1] == "png" and n_image < k):
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


def find_similar_files(s3_client, collection, query_emb: np.ndarray,k = 5):
    # Chroma expects list-of-lists for query_embeddings
    query_vector = query_emb.tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2000,
        include=["documents", "distances"]
    )

    print_top_k_files(results,s3_client,k)

    return results

def find_k_similars_by_file(s3_client, collection, doc, doc_type,k = 5):
    query_emb = None
    if doc_type == "image":
        query_emb = embed_image(preprocess_it, model_it,doc,device)

    elif doc_type == "text":
        query_emb = embed_text(tokenizer_it, model_it,doc,device)

    if query_emb is not None:
        results = find_similar_files(s3_client,collection,query_emb,k)
        return results
    return None
