# Importing useful dependencies
import io
import ast
import time
import logging
import requests
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

def load_huggingface_dataset(dataset_id: str, split: str, cache_dir: str | None):
    """
    Load a specific split from a Hugging Face dataset in non-streaming mode.

    dataset_id  : str   - The dataset identifier on Hugging Face Hub.
    split       : str   - The dataset split to load.
    cache_dir   : str   - Optional local path to store the downloaded dataset. If None, the default Hugging Face cache directory is used.

    """
    logger.info("Loading dataset 1: %s (split=%s)", dataset_id, split)
    t0 = time.perf_counter()
    ds = load_dataset(dataset_id, split=split, cache_dir=cache_dir)  # streaming removed
    logger.info("Loaded dataset 1 with %d rows in %.2fs", ds.num_rows, time.perf_counter() - t0)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(ds)}. "
                        f"Check the split name '{split}' for dataset '{dataset_id}'.")
    return ds


def upload_strings_separately(bucket_name, client, strings, path="temporal-landing/", prefix="text", limit: int = 1000, offset=1):
    """
    Upload each non-empty string as a separate text file to the bucket.
    Files are named: <path><prefix>_1.txt, <path><prefix>_2.txt, ...

    bucket_name : str   - Target S3/MinIO bucket.
    client      : obj   - S3-compatible client (e.g., boto3.client("s3")).
    strings     : list  - Iterable of strings; None/empty values are skipped.
    path        : str   - Folder prefix in the bucket (default: 'temporal-landing/').
    prefix      : str   - Filename prefix (default: 'text').
    limit       : int   - Maximum number of strings to upload.
    offset      : int   - starting index offset for filenames
    """

    # Ensure path ends with '/'
    if path and not path.endswith("/"):
        path += "/"

    summary = {
        "attempted": 0,
        "uploaded": 0,
        "skipped_empty": 0,
        "failed": 0,
    }

    logger.info(
        "Uploading strings: bucket=%s, path=%s, prefix=%s, limit=%d, offset=%d",
        bucket_name, path or "(root)", prefix, limit, offset
    )

    t0 = time.perf_counter()
    uploaded = 0
    for s in strings:
        summary["attempted"] += 1
        if not s:  # skip empty or None
            summary["skipped_empty"] += 1
            continue

        # stop if we reached the limit
        if limit is not None and uploaded >= limit:
            break

        uploaded += 1
        i = offset + uploaded  # numbering: offset+1, offset+2, ...
        object_name = f"{path}{prefix}_{i}.txt"

        try:
            client.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=io.BytesIO(s.encode("utf-8")),
                ContentType="text/plain",
            )
            summary["uploaded"] += 1
            logger.debug("Uploaded: s3://%s/%s", bucket_name, object_name)

        except Exception as e:
            summary["failed"] += 1
            logger.exception("Failed to upload %r -> s3://%s/%s: %s", s[:40] + ("..." if len(s) > 40 else ""), bucket_name, object_name, e)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Upload strings completed in %.2fs — attempted=%d, uploaded=%d, skipped_empty=%d failed=%d", elapsed,
        summary["attempted"], summary["uploaded"], summary["skipped_empty"],  summary["failed"]
    )

def extract_screenshot_urls(records, field_type = "comma"):
    """
    Clean and flatten screenshot fields into a list of image URLs.

    bucket_name : str       - Target S3/MinIO bucket.
    records     : iterable  - Iterable of raw screenshot field values
    field_type  : str       - Parsing mode:
                                - "comma"   : comma-separated URLs
                                - "vbar"    : vertical bar separated URLs with 'image' keys
    """
    urls = []

    for item in records:
        if not item:
            continue
        if field_type == "comma":
            # Split comma-separated URLs
            urls.extend([u.strip() for u in item.split(",") if u.strip()])

        elif field_type == "vbar":
            # Split string of dicts by '|' and extract image fields
            for rec in item.split("|"):
                rec = rec.strip()
                if not rec:
                    continue
                try:
                    d = ast.literal_eval(rec)
                    if isinstance(d, dict) and "image" in d:
                        urls.append(d["image"].strip())
                except Exception:
                    continue  # Skip invalid records

        else:
            raise ValueError(f"Unknown field_type: {field_type}")

    return urls


def upload_media_from_links(bucket_name, client, links,
                            path="temporal-landing/", prefix="image",
                            limit=None, start_index=1,
                            timeout=(10, 120),  # (connect, read)
                            retries=2, backoff=1.5):
    """
    Stream each URL and upload it directly to S3/MinIO, preserving file extensions.

    bucket_name : str                   - Target S3/MinIO bucket.
    client      : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    links       : iterable              - Iterable of URLs to upload.
    path        : str                   - Folder prefix in the bucket (default: 'temporal-landing/').
    prefix      : str                   - Filename prefix (default: 'image').
    limit       : int                   - Maximum number of URLs to upload.
    start_index : int                   - Starting index of URLs to upload.
    timeout     : tuple[float, float]   - (connect_timeout, read_timeout) for requests.
    retries     : int                   - Number of retry attempts on transient failures.
    backoff     : float                 - Backoff multiplier between retries.
    """

    # Ensure path ends with '/'
    if path and not path.endswith("/"):
        path += "/"
    uploaded = 0
    session = requests.Session()
    summary = {
        "attempted": 0,
        "uploaded": 0,
        "skipped_empty": 0,
        "skipped_http": 0,
        "skipped_timeout": 0,
        "failed": 0,
    }

    logger.info(
        "Starting upload from links: bucket=%s, path=%s, limit=%s, start_index=%s",
        bucket_name, path or "(root)", limit, start_index
    )
    t0 = time.perf_counter()
    try:
        for url in links:
            if not url:
                summary["skipped_empty"] += 1
                continue

            if limit is not None and uploaded >= limit:
                break

            summary["attempted"] += 1
            attempt = 0
            while True:
                try:
                    # Stream download to avoid loading full file in memory
                    with session.get(url, stream=True, timeout=timeout) as r:
                        # Raise for any HTTP error (404, 5xx, etc.)
                        r.raise_for_status()


                        ext = url.split('.')[-1].split('?')[0]  # get file extension
                        object_name = f"{path}{prefix}_{start_index}.{ext}"
                        start_index = start_index + 1

                        # Stream → S3
                        client.upload_fileobj(
                            Fileobj=r.raw,
                            Bucket=bucket_name,
                            Key=object_name,
                            ExtraArgs={"ContentType": f"{prefix}/{ext}"}
                        )

                        summary["uploaded"] += 1
                        uploaded += 1
                        logger.debug("Uploaded: %s -> s3://%s/%s", url, bucket_name, object_name)

                        break  # success, exit retry loop

                except requests.exceptions.Timeout:
                    attempt += 1

                    if attempt > retries:
                        summary["skipped_timeout"] += 1
                        logger.warning("Timeout (skipped after %d retries): %s", retries, url)
                        break

                    time.sleep(backoff ** attempt)

                except requests.exceptions.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    summary["skipped_http"] += 1

                    if status == 404:
                        logger.warning("HTTP 404 (skipped): %s", url)
                    else:
                        logger.warning("HTTP error %s for %s: %s", status, url, e)
                    break  # do not retry non-timeout HTTP errors by default

                except Exception as e:
                    attempt += 1

                    if attempt > retries:
                        summary["failed"] += 1
                        logger.exception("Failed after %d retries for %s: %s", retries, url, e)
                        break

                    sleep_s = backoff ** attempt
                    logger.warning("Error (retry %d/%d in %.1fs) for %s: %s", attempt, retries, sleep_s, url, e)
                    time.sleep(sleep_s)



    finally:
        session.close()

    elapsed = time.perf_counter() - t0
    logger.info(
        "Upload completed in %.2fs — attempted=%d, uploaded=%d, skipped(empty=%d, http=%d, timeout=%d), failed=%d", elapsed,
        summary["attempted"], summary["uploaded"], summary["skipped_empty"],
        summary["skipped_http"], summary["skipped_timeout"], summary["failed"]
    )