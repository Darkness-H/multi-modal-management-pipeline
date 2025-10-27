# Importing useful dependencies
import io
import boto3
import torch
import requests
import imageio
import chromadb
import open_clip
import numpy as np
from PIL import Image
import torch.nn as nn
from io import BytesIO
import ipywidgets as widgets
from IPython.display import display
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, AutoTokenizer, CLIPTextModelWithProjection

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load "ViT-B-16" model for images and texts
model_it, _, preprocess_it = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tokenizer_it = open_clip.get_tokenizer("ViT-B-16") # Tokenizer for texts
model_it.to(device)