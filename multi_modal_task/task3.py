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
# If you are having problems importing these functions from the transformers library, try executing them on Kaggle
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, CLIPVisionModelWithProjection, CLIPImageProcessor


