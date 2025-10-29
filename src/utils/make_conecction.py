# Importing useful dependencies
import logging
from typing import Optional, Dict

import boto3
import chromadb

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

def make_chromaDB_client(
    host: str = "localhost",
    port: int = 8000,
    use_ssl: bool = False,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    timeout: float = 30.0,
):
    """
    Build a ChromaDB HTTP client with host/port provided separately.

    host         : str    - Chroma server host, e.g., "localhost" or "chroma.example.com".
    port         : int    - Chroma server port, e.g., 8000 or 443.
    use_ssl      : bool   - Use HTTPS if True, HTTP if False. Default False.
    access_key   : str    - Optional: sent as "X-Access-Key" header.
    secret_key   : str    - Optional: sent as "X-Secret-Key" header.
    bearer_token : str    - Optional: sent as "Authorization: Bearer <token>".
    timeout      : float  - Optional: client timeout seconds (if supported by your chromadb version).

    Returns
    -------
    chromadb.HttpClient
    """
    # Compose optional headers
    headers: Dict[str, str] = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if access_key:
        headers["X-Access-Key"] = access_key
    if secret_key:
        headers["X-Secret-Key"] = secret_key

    # Create client (handle version differences in chromadb)
    try:
        client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=use_ssl,
            headers=headers or None,
            timeout=timeout,
        )
    except TypeError:
        # Fallback for older chromadb that doesn't accept ssl/headers/timeout
        client = chromadb.HttpClient(host=host, port=port)

    return client

