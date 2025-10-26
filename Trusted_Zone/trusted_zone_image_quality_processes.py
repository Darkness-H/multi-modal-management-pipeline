#%%
# Importing useful dependencies
import base64
import io
import logging
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def get_data_images(client, bucket, prefix=""):
    """
    List objects under `prefix`, filter to images, and return basic image metadata.

    For each image, downloads the object and extracts width, height, mode, channels,
    and aspect ratio using Pillow (PIL).

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    prefix          : str                   - Optional key prefix (acts like a folder path).
    """

    # Ensure path ends with '/'
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    data: List[Dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    t0=time.perf_counter()
    summary = {
        "scanned": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
    }
    etags=set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            duplicate = False
            key = obj["Key"]
            summary["scanned"] += 1
            # skip the folder itself

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
                # Download the image
                resp = client.get_object(Bucket=bucket, Key=key)
                body = resp["Body"].read()

            except Exception as e:
                summary["failed"] += 1
                logger.exception("Failed to process %s: %s", key, e)
                continue

            try:
                img = Image.open(io.BytesIO(body))
                width, height = img.size
                mode = img.mode
                n_channels = len(mode) if mode in ["RGB","RGBA","L","CMYK"] else None

            except Exception as e:
                summary["failed"] += 1
                logger.exception("Failed to open image %s: %s", key, e)
                continue

            data.append({
                "file_name": key,
                "file_size": obj['Size'] / 1024, # B -> KB
                "width": width,
                "height": height,
                "aspect_ratio": round(width/height, 2) if height > 0 else None, # proportional relationship between image width and height
                "mode": mode, # how pixel values are stored
                "channels": n_channels, # number of color channels in the image
                "duplicated": duplicate,
            })

            summary["succeeded"] += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Images data extraction completed in %.2fs — scanned=%d, succeeded=%d, skipped=%d, failed=%d",
        elapsed, summary["scanned"], summary["succeeded"],
        summary["skipped"], summary["failed"]
    )

    return data


def generate_data_quality_images(df):
    """
    Generate an automated image data quality report (HTML + charts).

    The function analyzes a dataset of image metadata (e.g., file size, width,
    height, aspect ratio, mode) and produces visual summaries and descriptive
    statistics to help assess dataset consistency and detect anomalies.

    df              : pandas.DataFrame      -  DataFrame containing per-image metadata. Expected columns include:
                                                - 'file_name'     : str  – Image key or file path.
                                                - 'file_size'     : float – File size in kilobytes (KB).
                                                - 'width'         : int   – Image width in pixels.
                                                - 'height'        : int   – Image height in pixels.
                                                - 'aspect_ratio'  : float – Width divided by height.
                                                - 'mode'          : str   – Color mode (e.g., RGB, RGBA, L, CMYK).
                                                - 'channels'      : int   - number of color channels in the image
                                                - 'duplicated'    : bool  - identifiers for duplicate detection.
    """


    num_images = df.shape[0]
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
    # Width vs Height of Images
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df, x="width", y="height", alpha=0.5, ax=ax)
    ax.set_title("Width vs Height")
    plots["width_height"] = fig_to_base64(fig)

    # Aspect Ratio Distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["aspect_ratio"].dropna(), bins=50, ax=ax)
    ax.set_title("Aspect Ratio Distribution (W/H)")
    ax.set_xlabel("Aspect Ratio")
    plots["aspect_ratio"] = fig_to_base64(fig)




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
        <div class="metric">Total images: <b>{num_images}</b></div>
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
        <img src="data:image/png;base64,{plots["size_dist"]}">
        <h3>Width vs Height</h3>
        <img src="data:image/png;base64,{plots["width_height"]}">
        <h3>Aspect Ratio Distribution</h3>
        <img src="data:image/png;base64,{plots["aspect_ratio"]}">
    </body>
    </html>
    """

    # Ensure output folder exists in the current working directory
    os.makedirs("reports", exist_ok=True)

    # Build output path under reports/
    out_path = os.path.join("reports", "image_quality_report.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report saved to %s", out_path)


def preprocess_image(client ,bucket, prefix="", target_size=(512, 512)):
    """
    Normalize and clean image data stored in an S3/MinIO bucket.

    The function analyzes and preprocesses all images under the given prefix:
    it generates a data quality report, removes duplicates and empty objects,
    converts images to RGB, resizes them to a fixed target size, normalizes
    pixel values, and re-uploads them to the same bucket.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    prefix          : str                   - Optional key prefix (acts like a folder path).
    target_size     : tuple[int, int]       - optional Target spatial resolution
    """

    data = get_data_images(client, bucket, prefix=prefix)
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if df.empty:
        raise ValueError("No image data to report.")
    generate_data_quality_images(df)

    summary = {
        "total_rows": int(df.shape[0]),
        "empty_deleted": 0,
        "duplicates_deleted": 0,
        "normalized": 0,
        "skipped_errors": 0,
    }
    t0 = time.perf_counter()

    logger.info(
        "Starting image preprocessing: rows=%d, target_size=%s, bucket=%s, prefix=%s",
        summary["total_rows"], target_size, bucket, prefix
    )
    for row in df.itertuples(index=False):

        # get image name
        key = row.file_name

        if not key:
            summary["skipped_errors"] += 1
            logger.warning("Row missing 'file_name'; skipping.")
            continue
        # remove empty or duplicated image
        if row.duplicated:
            client.delete_object(Bucket=bucket, Key=key)
            summary["duplicates_deleted"] += 1
            logger.debug(f"Deleted duplicate: {key}")
            continue




        try:
            # Get object metadata to check size
            head = client.head_object(Bucket=bucket, Key=key)
            if head['ContentLength'] == 0:
                # Delete the null image (size = 0)
                client.delete_object(Bucket=bucket, Key=key)
                summary["empty_deleted"] += 1
                logger.debug(f"Deleted empty object (size=0): {key}")
                continue
        except Exception as e:
            summary["skipped_errors"] += 1
            logger.error(f"head_object failed for {key}: {e}")
            continue

        try:
            # Download the image
            resp = client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"].read()

            # Load image with Pillow
            img = Image.open(io.BytesIO(body))

            # Convert images into RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')  # 3 channels

            # Resize image
            img = img.resize(target_size, Image.LANCZOS)  # Image.LANCZOS = image smoother

            # Convert to numpy array and normalize pixel values to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0

            # Convert back to PIL Image
            img_out = Image.fromarray((img_array * 255).astype(np.uint8))
            buf = io.BytesIO()
            img_out.save(buf, format="PNG")
            buf.seek(0)

            # overwrite the same key with PNG (content-type updated)
            client.upload_fileobj(
                Fileobj=buf,
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ContentType": "image/png"},
            )
            summary["normalized"] += 1
            logger.debug(f"Normalized: {key}")

        except UnidentifiedImageError:
            summary["skipped_errors"] += 1
            logger.error(f"Unreadable/corrupted image: {key}")
        except Exception as e:
            summary["skipped_errors"] += 1
            logger.error(f"Failed to normalize {key}: {e}")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Image preprocessing completed in %.2fs - total_image=%d, empty_deleted=%d, duplicates_deleted=%d, "
        "normalized=%d, errors=%d",
        elapsed,
        summary["total_rows"],
        summary["empty_deleted"],
        summary["duplicates_deleted"],
        summary["normalized"],
        summary["skipped_errors"],
    )


