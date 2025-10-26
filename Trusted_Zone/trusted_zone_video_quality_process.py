# Importing useful dependencies
import base64
from datetime import datetime
import io
import os
import time

import boto3
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def get_data_videos(client, bucket, prefix=""):
    """
    Collect basic video metadata under a given S3/MinIO prefix.

    The function downloads each video to a temporary file, probes it with OpenCV,
    and returns a list of records with key properties (resolution, fps, frames,
    channels, aspect ratio), including an exact-duplicate flag based on (etag, size).

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    prefix          : str                   - Optional key prefix (acts like a folder path).
    """
    data = []
    t0=time.perf_counter()
    summary = {
        "scanned": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
    }
    paginator = client.get_paginator("list_objects_v2")
    etags=set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):

            duplicate = False
            key = obj["Key"]
            summary["scanned"] += 1
            size = obj["Size"]
            etag = obj["ETag"].strip('"')
            if size == 0 and key.endswith("/"):
                summary["skipped"] += 1
                continue

            # duplication check
            sig = (etag, size)
            if sig in etags:
                duplicate = True
            else:
                etags.add(sig)

            try:
                # Download the video
                resp = client.get_object(Bucket=bucket, Key=key)
                body = resp["Body"].read()
            except Exception as e:
                summary["failed"] += 1
                logger.exception("Failed to process %s: %s", key, e)
                continue
            try:
                # Use a temporary file to read the video with OpenCV
                with open("temp_video_in.mp4", "wb") as f:
                    f.write(body)

                cv2.VideoCapture()

                cap = cv2.VideoCapture("temp_video_in.mp4")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                ret, frame = cap.read()
                if ret:
                    n_channels = frame.shape[2] if len(frame.shape) == 3 else 1
                    mode = 'BGR' if n_channels == 3 else 'GRAY'
                else:
                    n_channels = 0
                    mode = "N/A"

                cap.release()
                os.remove("temp_video_in.mp4")
            except Exception as e:
                summary["failed"] += 1
                logger.exception("Failed to open video %s: %s", key, e)
                continue

            data.append({
                "file_name": key,
                "file_size": obj['Size'] / 1024,  # B -> KB
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 2) if height > 0 else None,
                "fps": fps,
                "frames": frames,
                "mode": mode,
                "channels": n_channels,
                "duplicated": duplicate
            })
            summary["succeeded"] += 1
    elapsed = time.perf_counter() - t0
    logger.info(
        "Videos data extraction completed in %.2fs — scanned=%d, succeeded=%d, skipped=%d, failed=%d",
        elapsed, summary["scanned"], summary["succeeded"],
        summary["skipped"], summary["failed"]
    )
    return data,summary


def generate_data_quality_videos(df):
    """
    Generate an automated video data quality report (HTML + charts).

    The function analyzes a dataset of video metadata (e.g., file size, width,
    height, aspect ratio, frame rate, frame count, channels) and produces
    visual summaries and descriptive statistics to assess dataset consistency
    and detect anomalies (e.g., unreadable files, extreme aspect ratios,
    tiny/oversized resolutions, duplicate assets).

    df              : pandas.DataFrame      -  DataFrame containing per-image metadata. Expected columns include:
                                                - 'file_name'     : str  – Image key or file path.
                                                - 'file_size'     : float – File size in kilobytes (KB).
                                                - 'width'         : int   – Image width in pixels.
                                                - 'height'        : int   – Image height in pixels.
                                                - 'aspect_ratio'  : float – Width divided by height.
                                                -  'fps'          : float  – Frames per second.
                                                -  'frames'       : int    – Total frame count.
                                                - 'mode'          : str   – Color mode (e.g., RGB, RGBA, L, CMYK).
                                                - 'channels'      : int   - number of color channels in the image
                                                - 'duplicated'    : bool  - identifiers for duplicate detection.
    """


    num_videos = df.shape[0]
    num_duplicated = df["duplicated"].sum()
    head_html = df.head(10).to_html(border = 0)
    # Have a quick summary of the data
    desc_html = df.describe(include="all").to_html(border=0)

    # Unique values for mode
    unique_modes  = pd.unique(df['mode'])
    modes_str = ", ".join(map(str, unique_modes))  # e.g., "RGB, RGBA, L"
    modes_count = len(unique_modes)
    # charts generation
    plots = {}
    def fig_to_base64(figure):
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight")
        plt.close(figure)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # File size distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["file_size"], bins=50, ax=ax)
    ax.set_title("File Size Distribution (KB)")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    plots["size_dist"] = fig_to_base64(fig)

    # Width vs Height of Videos
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df, x="width", y="height", alpha=0.5, ax=ax)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Width vs Height of Videos")
    plots["width_height"] = fig_to_base64(fig)

    # Aspect Ratio Distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["aspect_ratio"].dropna(), bins=50, ax=ax)
    ax.set_title("Aspect Ratio Distribution")
    ax.set_xlabel("Aspect Ratio (W/H)")
    plots["aspect_ratio"] = fig_to_base64(fig)

    # Fps distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["fps"].dropna(), bins=50, ax=ax)
    ax.set_xlabel("Frames Per Second (FPS)")
    ax.set_title("FPS Distribution")
    plots["fps"] = fig_to_base64(fig)

    # Total frames distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["frames"].dropna(), bins=50, ax=ax)
    ax.set_xlabel("Total Frames")
    ax.set_title("Total Frames Distribution")
    plots["frames"] = fig_to_base64(fig)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Image Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ color: #333; }}
            img {{ max-width: 700px; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
            .metric {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h1>📊 Image Data Quality Report</h1>
        <p>Generated on: <b>{now}</b></p>
        <hr>
        <h2>Summary</h2>
        <div class="metric">Total videos: <b>{num_videos}</b></div>
        <div class="metric">Row-level duplicates: <b>{num_duplicated}</b></div>
        <div class="metric">Average file size: <b>{df["file_size"].mean():.2f} KB</b></div>
        <div class="metric">Resolution (avg): <b>{df["width"].mean():.0f} × {df["height"].mean():.0f}</b></div>
        <div class="metric">Unique modes: <b>{modes_count}</b> ({modes_str})</div>
        <h2>DataFrame head()</h2>
        {head_html}
        <h2>DataFrame describe()</h2>
        {desc_html}
        <hr>
        <h2>Distributions</h2>
        <h3>File Size Distribution</h3>
        <img src="data:video/png;base64,{plots["size_dist"]}">
        <h3>Width vs Height</h3>
        <img src="data:video/png;base64,{plots["width_height"]}">
        <h3>Aspect Ratio Distribution</h3>
        <img src="data:video/png;base64,{plots["aspect_ratio"]}">
        <h3>Fps distribution</h3>
        <img src="data:video/png;base64,{plots["fps"]}">
        <h3>Total frames distribution</h3>
        <img src="data:video/png;base64,{plots["frames"]}">
    </body>
    </html>
    """

    # Ensure output folder exists in the current working directory
    os.makedirs("reports", exist_ok=True)

    # Build output path under reports/
    out_path = os.path.join("reports", "video_quality_report.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report saved to %s", out_path)

def preprocess_video(client, bucket, prefix="", target_fps = 1, target_size=(224, 224)):
    """
    Normalize and clean video data stored in an S3/MinIO bucket.

    The function analyzes and preprocesses all videos under the given prefix:
    it generates a data quality report, removes corrupted/empty objects and near-duplicate frames,
    extracts frames at a fixed target FPS, converts frames to RGB, resizes them to a fixed target size,
    normalizes pixel values, and re-uploads the outputs to the same bucket.


    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    prefix          : str                   - Optional key prefix (acts like a folder path).
    target_size     : tuple[int, int]       - optional Target spatial resolution
    """
    data,summary = get_data_videos(client, bucket, prefix=prefix)
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if df.empty:
        raise ValueError("No image data to report.")
    generate_data_quality_videos(df)

    summary = {
        "total_rows": int(df.shape[0]),
        "empty_deleted": 0,
        "duplicates_deleted": 0,
        "normalized": 0,
        "skipped_errors": 0,
    }
    t0 = time.perf_counter()

    for row in df.itertuples(index=False):
        # get video name
        key = row.file_name