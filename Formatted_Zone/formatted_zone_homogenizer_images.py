# Importing useful dependencies
import io
import logging
import os
import time

from PIL import Image

logger = logging.getLogger(__name__)
def convert_images_to_png(client, bucket, prefix=""):
    """
    Scan objects under `s3://{bucket}/{prefix}`, convert non-PNG images to PNG,
    upload them as `.png` (same basename), then delete the original files.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - S3/MinIO bucket name to process.
    prefix          : str                   - Optional key prefix (acts like a folder path).
    """

    # Ensure path ends with '/'
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    summary = {
        "scanned": 0,
        "converted": 0,
        "skipped_png": 0,
        "skipped_folders": 0,
        "failed": 0,
    }
    logger.info("Starting PNG conversion: s3://%s/%s", bucket, prefix or "(root)")
    t0 = time.perf_counter()

    paginator = client.get_paginator("list_objects_v2")  # paginate to avoid loading all keys at once
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        #`Contents` may be missing on empty pages, so we .get(..., []) safely
        for obj in page.get("Contents", []):
            summary["scanned"] += 1
            key = obj["Key"]
            # Skip folder markers: zero-byte objects whose keys end with '/'
            if obj["Size"] == 0 and key.endswith("/"):
                summary["skipped_folders"] += 1
                continue

            # Skip images that are already PNG (case-insensitive extension check)
            if os.path.splitext(key)[1].lower() == ".png":
                summary["skipped_png"] += 1
                continue

            # Compute the new key by replacing the extension with '.png'
            new_key = os.path.splitext(key)[0] + ".png"

            try:
                # Download the source object bytes into memory
                resp = client.get_object(Bucket=bucket, Key=key)
                body = resp["Body"].read()

                # Open the image from bytes and convert to PNG
                img = Image.open(io.BytesIO(body))
                buf = io.BytesIO()

                # Note: Pillow auto-converts mode as needed when saving PNG.
                # If source has alpha or palette, Pillow will handle appropriately.

                img.save(buf, format="PNG")
                buf.seek(0)  # reset buffer pointer before upload

                # Upload the converted PNG back under the new key
                client.upload_fileobj(
                    buf,
                    Bucket=bucket,
                    Key=new_key,
                    ExtraArgs={"ContentType": "image/png"}  # set correct MIME type
                )

                # Delete the original non-PNG object to "replace" it
                client.delete_object(Bucket=bucket, Key=key)
                summary["converted"] += 1
                logger.debug("Converted: %s -> %s ",
                             key, new_key)

            except Exception as e:
                summary["failed"] += 1
                logger.exception("Failed to convert %s: %s", key, e)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Images conversion completed in %.2fs — scanned=%d, converted=%d, skipped(folders=%d, png=%d), failed=%d",
        elapsed, summary["scanned"], summary["converted"],
        summary["skipped_folders"], summary["skipped_png"], summary["failed"]
    )