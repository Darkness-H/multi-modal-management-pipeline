# Importing useful dependencies
import time

from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

def replicate_bucket(client, src_bucket: str, dest_bucket: str, src_prefix: str = "", dest_prefix: str = "") -> None:
    """
    Copy all objects from `s3://src_bucket/src_prefix` into `s3://dest_bucket/dest_prefix`.
    Creates the destination bucket if it does not exist.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    src_bucket      : str                   - source S3/MinIO bucket.
    dest_bucket     : str                   - Target S3/MinIO bucket.
    src_prefix      : str                   - Source folder/prefix to scan and move from (e.g. "temporal-landing/"). Directory markers like `prefix/` or zero-byte `.keep` are skipped.
    dest_prefix     : str                   - Destination root prefix (e.g. "persistent-landing/"). Classified subfolders are created under this root: `texts/`, `images/`, `videos/`.

    """

    # Normalize prefixes: either empty "" or end with "/"
    if src_prefix and not src_prefix.endswith("/"):
        src_prefix += "/"
    if dest_prefix and not dest_prefix.endswith("/"):
        dest_prefix += "/"

    # Ensure destination bucket exists
    try:
        client.head_bucket(Bucket=dest_bucket)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            # MinIO typically doesn't require LocationConstraint
            client.create_bucket(Bucket=dest_bucket)
            logging.info("Created bucket: %s", dest_bucket)
        else:
            raise

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=src_bucket, Prefix=src_prefix)

    summary = {"copied": 0, "skipped_folders": 0, "failed": 0, "scanned": 0}
    t0 = time.perf_counter()
    logger.info(
        "Replicating objects: %s/%s -> %s/%s",
        src_bucket, src_prefix or "(root)", dest_bucket, dest_prefix or "(root)"
    )
    for page in pages:
        for obj in page.get("Contents", []):
            summary["scanned"] += 1
            key = obj["Key"]
            size = obj["Size"]

            # Skip folder markers
            if size == 0 and key.endswith("/"):
                summary["skipped_folders"] += 1
                continue

            # Compute relative part safely
            if src_prefix and key.startswith(src_prefix):
                relative = key[len(src_prefix):]
            else:
                # If no src_prefix provided or mismatch, treat full key as relative
                relative = key

            new_key = f"{dest_prefix}{relative}"

            # Copy
            try:
                client.copy_object(
                    Bucket=dest_bucket,
                    Key=new_key,
                    CopySource={"Bucket": src_bucket, "Key": key},
                )
                summary["copied"] += 1
                logger.debug("Copied: %s/%s -> %s/%s", src_bucket, key, dest_bucket, new_key)

            except ClientError as e:
                summary["failed"] += 1
                logger.exception("Failed to copy %s/%s -> %s/%s: %s", src_bucket, key, dest_bucket, new_key, e)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Replication completed in %.2fs — scanned=%d, copied=%d, skipped_folders=%d, failed=%d",
            elapsed, summary["scanned"], summary["copied"], summary["skipped_folders"], summary["failed"]
        )