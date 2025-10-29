# Importing useful dependencies
import time
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)
def classify_object_by_head(client, bucket, key, default="others"):
    """
    Classify an S3 object as 'texts' | 'images' | 'videos' by ContentType.
    Falls back to file extension if ContentType is missing/unknown.

    client      : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket      : str                   - Target S3/MinIO bucket.
    key         : str                   - Object key (path) within the bucket.
    default     : str                   - optional
                                          Fallback class to return when detection fails or type is not recognized.
                                          Defaults to "others".
    """

    try:
        # 1) Ask S3 for ContentType (may include charset)
        head = client.head_object(Bucket=bucket, Key=key)
        ctype = (head.get("ContentType") or "").split(";", 1)[0].strip().lower()
    except ClientError as e:
        # No such key or access denied → return default
        return default
    except Exception:
        return default

    # 2) Primary decision by MIME prefix
    if ctype.startswith("text/"):
        return "texts"
    if ctype.startswith("image/"):
        return "images"
    if ctype.startswith("video/"):
        return "videos"

def move_files(
               client,
               bucket: str,
               source_prefix: str = "temporal-landing/",
               dest_prefix: str = "persistent-landing/",
               allowed: set[str]= {"texts", "images", "videos"}):
    """
    Move objects from `source_prefix` to `dest_prefix` by class:
    - classify each object (texts/images/videos/…),
    - build timestamped filename (ms since epoch [+ uuid]),
    - copy to `<dest_prefix>/<class>/`,
    - delete original.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    bucket          : str                   - Target S3/MinIO bucket.
    source_prefix   : str                   - Source folder/prefix to scan and move from (e.g. "temporal-landing/"). Directory markers like `prefix/` or zero-byte `.keep` are skipped.
    dest_prefix     : str                   - Destination root prefix (e.g. "persistent-landing/"). Classified subfolders are created under this root: `texts/`, `images/`, `videos/`.
    allowed         : set[str]              - optional
                                              Whitelist of class names to move (case-insensitive). Non-matching classes are skipped. Default is {"texts", "images", "videos"}.
    """

    # Ensure path ends with '/'
    if source_prefix and not source_prefix.endswith("/"):
        source_prefix += "/"
    if dest_prefix and not dest_prefix.endswith("/"):
        dest_prefix += "/"

    summary = {
        "moved": 0,
        "skipped_folders": 0,
        "skipped_disallowed": 0,
        "skipped_classify_error": 0,
        "failed": 0,
        "scanned": 0,
    }

    t0 = time.perf_counter()
    logger.info(
        "Moving files: bucket=%s, src=%s, dest=%s, allowed=%s",
        bucket, source_prefix or "(root)", dest_prefix or "(root)", sorted(allowed)
    )

    paginator = client.get_paginator("list_objects_v2")  # It returns objects in pages and not all at once.
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            summary["scanned"] += 1
            src_key = obj["Key"]

            if obj['Size'] == 0 and src_key.endswith("/"):
                summary["skipped_folders"] += 1
                continue

            # classify
            try:
                category = classify_object_by_head(client, bucket, src_key)
            except Exception as e:
                summary["skipped_classify_error"] += 1
                logger.warning("Classification failed for %s: %s", src_key, e)
                continue
            # we don't need data with type differents texts,videos and imaages
            if category not in allowed:
                summary["skipped_disallowed"] += 1
                logger.warning("Disallowed category (%s), skip: %s", category, src_key)
                continue
            # get file extension
            ext = src_key.split('.')[-1].split('?')[0]

            # new filename = timestamp + original extension
            ts = int(time.time() * 1000)  # milliseconds
            new_filename = f"{category[0:-1]}_{ts}.{ext}"

            # build destination key
            dest_key = f"{dest_prefix}{category}/{new_filename}"

            # copy then delete
            try:
                client.copy_object(Bucket=bucket,
                                   CopySource={"Bucket": bucket, "Key": src_key},
                                   Key=dest_key)
                client.delete_object(Bucket=bucket, Key=src_key)
                summary["moved"] += 1
                logger.debug("Moved: %s -> %s", src_key, dest_key)
            except ClientError as e:
                summary["failed"] += 1
                logger.exception("Failed to move %s -> %s: %s", src_key, dest_key, e)

    elapsed = time.perf_counter() - t0
    logger.info(
        "move_files completed in %.2fs — scanned=%d, moved=%d, skipped(folders=%d, disallowed=%d, classify_err=%d), failed=%d",
        elapsed,
        summary["scanned"],
        summary["moved"],
        summary["skipped_folders"],
        summary["skipped_disallowed"],
        summary["skipped_classify_error"],
        summary["failed"],
    )