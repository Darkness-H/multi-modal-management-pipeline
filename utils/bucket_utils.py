# Importing useful dependencies
import logging
import random
import time
from typing import Iterable, Optional, List
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)

def ensure_bucket(s3, bucket_name: str) -> None:
    """
    Ensure the bucket exists. Create it if it does not.
    Uses HeadBucket to avoid fetching the full bucket list.
    """

    try:
        s3.head_bucket(Bucket=bucket_name)
        logger.info("Bucket '%s' already exists.", bucket_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            # For MinIO, LocationConstraint is typically not required.
            # For AWS S3 in some regions you must pass CreateBucketConfiguration.
            s3.create_bucket(Bucket=bucket_name)
            logger.info("Created bucket: %s", bucket_name)
        else:
            # Re-raise unexpected errors (permissions, network issues, etc.)
            raise

def ensure_prefixes(s3, bucket_name: str, prefixes: Iterable[str]) -> None:
    """
    Ensure a set of folder-like prefixes exist inside the bucket.
    S3/MinIO has no real directories; we create a zero-byte object with a trailing slash.
    This operation is idempotent.
    """
    for p in prefixes:
        key = p if p.endswith("/") else (p + "/")
        try:
            # Creating an empty object named 'prefix/' makes most UIs show it as a folder.
            s3.put_object(Bucket=bucket_name, Key=key)
            logger.info("Ensured prefix: %s", key)
        except ClientError as e:
            logger.error("Failed to ensure prefix %s: %s", key, e)
            raise

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


def list_s3_keys(
    client,
    bucket: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    max_keys: Optional[int] = None
) -> List[str]:
    """
    List object keys in an S3-compatible bucket with optional prefix/suffix filtering.
    Uses paginator to handle large listings.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - source S3/MinIO bucket.
    prefix          : str                   - only include keys that start with this prefix
    suffix          : str                   - only include keys that end with this suffix (e.g., ".txt")
    max_keys        : str                   - if set, stop after collecting this many keys

    """
    paginator = client.get_paginator("list_objects_v2")
    pagination_cfg = {"Bucket": bucket}
    if prefix:
        pagination_cfg["Prefix"] = prefix

    keys = []
    for page in paginator.paginate(**pagination_cfg):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            # skip "folders" (zero-sized placeholders that end with '/')
            if key.endswith("/"):
                continue
            if suffix and not key.endswith(suffix):
                continue
            keys.append(key)
            if max_keys and len(keys) >= max_keys:
                return keys
    return keys


def get_random_s3_key(
    s3_client,
    bucket: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None
) -> str:
    """
    Return a single random key from an S3-compatible bucket,
    constrained by optional prefix/suffix.
    """
    keys = list_s3_keys(s3_client, bucket, prefix=prefix, suffix=suffix)
    if not keys:
        raise FileNotFoundError(f"No objects found in '{bucket}' (prefix={prefix!r}, suffix={suffix!r}).")
    return random.choice(keys)
