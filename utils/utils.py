# Importing useful dependencies
from botocore.exceptions import ClientError
import logging


def replicate_bucket(client, src_bucket: str, dest_bucket: str, src_prefix: str = "", dest_prefix: str = "") -> None:
    """
    Copy all objects from `s3://src_bucket/src_prefix` into `s3://dest_bucket/dest_prefix`.
    Creates the destination bucket if it does not exist.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    src_bucket      : str                   - source S3/MinIO bucket.
    dest_bucket     : str                   - Target S3/MinIO bucket.
    source_prefix   : str                   - Source folder/prefix to scan and move from (e.g. "temporal-landing/"). Directory markers like `prefix/` or zero-byte `.keep` are skipped.
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

    copied = 0
    skipped = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]

            # Skip folder markers
            if size == 0 and key.endswith("/"):
                skipped += 1
                continue

            # Compute relative part safely
            if src_prefix and key.startswith(src_prefix):
                relative = key[len(src_prefix):]
            else:
                # If no src_prefix provided or mismatch, treat full key as relative
                relative = key

            new_key = f"{dest_prefix}{relative}"

            # Copy
            copy_source = {"Bucket": src_bucket, "Key": key}
            client.copy_object(Bucket=dest_bucket, Key=new_key, CopySource=copy_source)
            print(f"Copied: {src_bucket}/{key} -> {dest_bucket}/{new_key}")
            copied += 1

    print(f"Done. Copied: {copied}, skipped folder markers: {skipped}")