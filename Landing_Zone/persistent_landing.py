# Importing useful dependencies
import time
import boto3
from botocore.exceptions import ClientError
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
    paginator = client.get_paginator("list_objects_v2")  # It returns objects in pages and not all at once.
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):

            src_key = obj["Key"]

            if obj['Size'] == 0 and src_key.endswith("/"):
                continue

            # classify
            category = classify_object_by_head(client, bucket, src_key)
            # we don't need data with type differents texts,videos and imaages
            if category not in allowed:
                continue
            # get file extension
            ext = src_key.split('.')[-1].split('?')[0]

            # new filename = timestamp + original extension
            ts = int(time.time() * 1000)  # milliseconds
            new_filename = f"{category[0:-1]}_{ts}.{ext}"

            # build destination key
            dest_key = f"{dest_prefix}{category}/{new_filename}"

            # copy then delete
            client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dest_key)
            client.delete_object(Bucket=bucket, Key=src_key)

            print(f"Moved: {src_key} -> {dest_key}")