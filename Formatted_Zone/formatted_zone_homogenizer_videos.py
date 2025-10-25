# Importing useful dependencies
import logging
import os
import time

import boto3
import warnings
import tempfile
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)
# The following line ignores any warnings MoviePy would normally print (like those ffmpeg frame read errors) just won’t show up.
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")

def convert_single_video(client, bucket: str, key: str) -> bool:
    """
    Convert a single non-MP4 video in S3 to MP4 format, upload it back, and delete the original.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    key          : str                      - Object key (path) within the bucket.
    """
    ext = os.path.splitext(key)[1].lower()

    new_key = os.path.splitext(key)[0] + ".mp4"
    tmp_in_path = tmp_out_path = None
    t_start = time.perf_counter()

    try:
        # Download original video
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
            resp = client.get_object(Bucket=bucket, Key=key)
            tmp_in.write(resp["Body"].read())
            tmp_in.flush()
            tmp_in_path = tmp_in.name

        # Prepare output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            tmp_out_path = tmp_out.name

        # Convert video
        clip = VideoFileClip(tmp_in_path)
        clip.write_videofile(tmp_out_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()

        # Upload MP4 to S3
        with open(tmp_out_path, "rb") as f:
            client.upload_fileobj(f, Bucket=bucket, Key=new_key, ExtraArgs={"ContentType": "video/mp4"})

        # Delete original
        client.delete_object(Bucket=bucket, Key=key)

        logger.debug("Converted: %s -> %s (%.2fs)", key, new_key, time.perf_counter() - t_start)
        return True

    except Exception as e:
        logger.warning("Failed to convert %s: %s", key, e)
        return False

    finally:
        # Clean up temp files
        for p in (tmp_in_path, tmp_out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def convert_videos_to_mp4(client, bucket: str, prefix: str = ""):
    """
    Scan a bucket for non-MP4 videos and convert them to MP4 using MoviePy.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    prefix          : str                   - Optional key prefix (acts like a folder path).
    """

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    summary = {
        "scanned": 0,
        "converted": 0,
        "skipped_folders": 0,
        "skipped_mp4": 0,
        "failed": 0,
    }

    t0 = time.perf_counter()
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            summary["scanned"] += 1
            key = obj["Key"]

            # Skip folder markers
            if obj["Size"] == 0 and key.endswith("/"):
                summary["skipped_folders"] += 1
                continue

            ext = os.path.splitext(key)[1].lower()
            if ext == ".mp4":
                summary["skipped_mp4"] += 1
                continue

            # Process conversion
            success = convert_single_video(client, bucket, key)
            if success:
                summary["converted"] += 1
            else:
                summary["failed"] += 1


    elapsed = time.perf_counter() - t0

    logger.info(
        "Conversion completed in %.2fs — scanned=%d, converted=%d, skipped(folders=%d, mp4=%d), failed=%d",
        elapsed,
        summary["scanned"],
        summary["converted"],
        summary["skipped_folders"],
        summary["skipped_mp4"],
        summary["failed"],
    )







