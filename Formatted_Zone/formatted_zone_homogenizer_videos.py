# Importing useful dependencies
import os
import boto3
import warnings
import tempfile
from moviepy import VideoFileClip

# The following line ignores any warnings MoviePy would normally print (like those ffmpeg frame read errors) just won’t show up.
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")
