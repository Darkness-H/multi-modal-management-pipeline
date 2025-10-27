# Importing useful dependencies
import open_clip
import torch

import numpy as np

from IPython.display import display
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from exploitation_zone.utils_exploitation.getter import get_text, get_image,get_video
from exploitation_zone.utils_exploitation.embeddings import embed_text,embed_image,embed_video



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

    print(f"Top {top_k} similar {collection.name}:")
    for rank, (doc_id, doc, dist) in enumerate(zip(ids, docs, dists), start=1):
        print(f"{rank}. id={doc_id}, distance={dist:.4f}")
        print(f"   document: {doc}")
        temp_path = doc.split("/")[-1]
        if (collection.name == "images"):
            display(get_image(s3_client,"trusted-zone", f"{collection.name}/{temp_path}"))
        elif (collection.name == "texts"):
            print("\n" + get_text(s3_client,"trusted-zone", f"{collection.name}/{temp_path}") + "\n")
        elif (collection.name == "videos"):
            frames = get_video(s3_client,"trusted-zone", f"{collection.name}/{temp_path}")
            for frame in frames:
                display(frame)
    return results


# Just in case our device has gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
model_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai",quick_gelu=True)
tokenizer = open_clip.get_tokenizer("ViT-B-16") # Tokenizer for texts
model_text.to(device)

def get_similar_text(s3_client, colleccion,text):
    query_embed = embed_text(tokenizer, model_text, text, device)
    res_text = find_similar_files(s3_client, colleccion, query_embed, top_k=5)
    return res_text

# Load model + preprocessing used when creating the image embeddings
model_image, _, preprocess_image = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai",quick_gelu=True)
model_image.to(device)

def get_similar_image(s3_client, colleccion,image):
    query_embed = embed_image(preprocess_image, model_image, image, device)
    res_image = find_similar_files(s3_client, colleccion, query_embed, top_k=5)
    return res_image


preprocess_video = CLIPImageProcessor.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model_video = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model_video.to(device)

def get_similar_video(s3_client, colleccion,video_frames):
    query_embed = embed_video(preprocess_video, model_video, video_frames, device)
    res_video = find_similar_files(s3_client, colleccion, query_embed, top_k=5)
    return res_video
