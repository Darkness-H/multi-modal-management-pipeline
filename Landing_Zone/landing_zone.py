# Importing useful dependencies
import logging
from typing import Iterable
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def make_s3_client(endpoint: str, access_key: str, secret_key: str):
    """
    Create an S3-compatible client (works with MinIO).
    """
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return s3

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

