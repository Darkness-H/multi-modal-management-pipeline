# Importing useful dependencies
import io
import os

import boto3
import torch
import imageio
import cv2 # for reading video frames
import chromadb
import requests
import open_clip
import numpy as np
from PIL import Image
from io import BytesIO
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
# If you are having problems importing these functions from the transformers library, try executing them on Kaggle
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, CLIPVisionModelWithProjection, CLIPImageProcessor

from exploitation_zone.exploitation_zone_image_embeddings import get_image
from exploitation_zone.utils_exploitation.embeddings import embed_text, embed_image
from exploitation_zone.utils_exploitation.getter import get_text

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load "ViT-B-16" model for images and texts
model_it, _, preprocess_it = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tokenizer_it = open_clip.get_tokenizer("ViT-B-16") # Tokenizer for texts
model_it.to(device)


# We will use this model as the answerer of our RAG system.
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

# Load model + processor
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
processor = AutoProcessor.from_pretrained(model_id)


def get_multimodal_prompt(description, user_query):
    multimodal_prompt = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a Game Recommendation Assistant, an expert in games."},
                {"type": "text", "text": "The user will ask you to recommend a game based on the user's request and an image."},
                {"type": "text", "text": "Your task is to generate a game description of similar type."},
                {"type": "text", "text": "You can use the following description and image as examples:"},
                {"type": "text", "text": description},
                {"type": "image"}, # <- first placeholder for images
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Generate a game description based on this request:\n{user_query}"},
                {"type": "image"} # <- second placeholder for images
            ],
        },
    ]

    return multimodal_prompt


def prepare_prompt(client, collection,query = "",image = None):
    retrieved_des = ""
    if query:
        query_emb = embed_text(tokenizer_it,model_it,query,device)
        query_vector = query_emb.tolist()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=["documents", "distances"]
        )
        retrieved_des = get_text(client,"trusted-zone", "texts/" + results['documents'][0][0].split("/")[-1])

    multimodal_prompt = get_multimodal_prompt(retrieved_des, query)

    images_list = None
    if image:
        image_emb = embed_image(preprocess_it,model_it,image,device)
        image_vector = image_emb.tolist()

        results = collection.query(
            query_embeddings=[image_vector],
            n_results=1,
            include=["documents", "distances"]
        )
        retrieved_image = get_image(client,"trusted-zone", "images/" + results['documents'][0][0].split("/")[-1])

        images_list = [retrieved_image,image]

    return multimodal_prompt,images_list

def get_recommendation(client, collection,query = None,image = None):
    multimodal_prompt,images_list = prepare_prompt(client,collection,query, image)
    prompt = processor.apply_chat_template(multimodal_prompt, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        images=images_list,  # must match number of placeholders
        return_tensors="pt"
    ).to(device, torch.float16)

    # Generate answer and decode
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded = processor.decode(output[0], skip_special_tokens=True)

    # Clean out possible prompt leftovers like "assistant" or special tokens
    clean_output = decoded.split("assistant\n")[-1].strip()
    clean_output = clean_output.replace(processor.image_token, "").strip()

    print("\n🎮 Model Recommendation:")
    print(clean_output)
    print("\nHere you have an image of the game: ")




