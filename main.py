import argparse
import logging
from dataclasses import dataclass
from typing import Optional, List

from multi_modal_task.task3 import get_recommendation
from utils.bucket_utils import replicate_bucket,ensure_bucket,ensure_prefixes
from utils.make_conecction import make_s3_client, make_chromaDB_client
from utils.collection_utils import create_chroma_collection

from Landing_Zone.temporal_landing import load_huggingface_dataset, upload_strings_separately,upload_media_from_links
from Landing_Zone.persistent_landing import move_files

from Formatted_Zone.formatted_zone_homogenizer_images import convert_images_to_png
from Formatted_Zone.formatted_zone_homogenizer_texts import convert_documents_to_txt
from Formatted_Zone.formatted_zone_homogenizer_videos import convert_videos_to_mp4

from Trusted_Zone.trusted_zone_image_quality_processes import preprocess_image
from Trusted_Zone.trusted_zone_video_quality_process import preprocess_video
from Trusted_Zone.trusted_zone_text_quality_processes import clean_text

from exploitation_zone.exploitation_zone_image_embeddings import images_to_embeddings, get_image_model
from exploitation_zone.exploitation_zone_text_embeddings import get_text_model, texts_to_embeddings
from exploitation_zone.utils_exploitation.getter import get_text, get_image, get_video
from exploitation_zone.exploitation_zone_video_embeddings import videos_to_embeddings, get_video_model
from exploitation_zone.embeddings_combination import combining_image_text

from multi_modal_task.task1 import get_similar_text, get_similar_image, get_similar_video
from multi_modal_task.task2 import find_k_similars_by_file

# ---- Dataclasses to carry structured args ----
@dataclass
class MinioArgs:
    endpoint: str
    access_key: str
    secret_key: str
    log_level: str

@dataclass
class LandingArgs:
    bucket: str
    prefixes: List[str]

@dataclass
class TemporalArgs:
    limit: int
    video_limit: int


@dataclass
class ChromaArgs:
    host: str
    port: int
    use_ssl: bool
    access_key: Optional[str]
    secret_key: Optional[str]
    bearer_token: Optional[str]
    timeout: float

@dataclass
class AllArgs:
    minio: MinioArgs
    landing: LandingArgs
    temporal: TemporalArgs
    chroma: ChromaArgs


def parse_all_args() -> AllArgs:
    """
    Parse a single CLI that contains args for:
    - MinIO credentials/init
    - Landing zone init
    - ChromaDB connection

    Returns a structured AllArgs with sub-namespaces.
    """
    parser = argparse.ArgumentParser(
        description="All-in-one CLI for MinIO init, landing, temporal dataset, and ChromaDB connection."
    )

    # -------- MinIO group --------
    g_minio = parser.add_argument_group("MinIO / S3")
    g_minio.add_argument("--minio-endpoint", default="http://127.0.0.1:9000",
                         help="MinIO/S3 endpoint URL.")
    g_minio.add_argument("--minio-access-key", default="minioadmin",
                         help="Access key (username) for MinIO/S3.")
    g_minio.add_argument("--minio-secret-key", default="minioadmin",
                         help="Secret key (password) for MinIO/S3.")
    g_minio.add_argument("--minio-log-level", default="INFO",
                         help="Logging level: DEBUG|INFO|WARNING|ERROR.")

    # -------- Landing group --------
    g_land = parser.add_argument_group("Landing Zone")
    g_land.add_argument("--landing-bucket", default="landing-zone",
                        help="Bucket name to ensure/create.")
    g_land.add_argument("--landing-prefixes", nargs="*", default=["temporal-landing/", "persistent-landing/"],
                        help="Folder-like prefixes to ensure under the bucket (trailing slash optional).")

    # -------- Temporal Landing group --------
    g_temp = parser.add_argument_group("Temporal Landing")
    g_temp.add_argument("--limit", default=500,
                        help= "maxim number of data extracted from one dataset -1 for no limit")
    g_temp.add_argument("--video-limit", default=10, help="maxim number of video extracted from one dataset")
    # -------- ChromaDB group --------
    g_chroma = parser.add_argument_group("ChromaDB")
    g_chroma.add_argument("--chroma-host", default="localhost", help="Chroma server host.")
    g_chroma.add_argument("--chroma-port", type=int, default=8000, help="Chroma server port.")
    g_chroma.add_argument("--chroma-use-ssl", action="store_true", help="Use HTTPS if set.")
    g_chroma.add_argument("--chroma-access-key", default=None, help="Optional X-Access-Key header.")
    g_chroma.add_argument("--chroma-secret-key", default=None, help="Optional X-Secret-Key header.")
    g_chroma.add_argument("--chroma-bearer-token", default=None, help="Optional Authorization: Bearer <token>.")
    g_chroma.add_argument("--chroma-timeout", type=float, default=30.0, help="Client timeout seconds.")

    ns = parser.parse_args()

    minio = MinioArgs(
        endpoint=ns.minio_endpoint,
        access_key=ns.minio_access_key,
        secret_key=ns.minio_secret_key,
        log_level=ns.minio_log_level,
    )

    landing = LandingArgs(
        bucket=ns.landing_bucket,
        prefixes=list(ns.landing_prefixes or []),
    )

    temporal = TemporalArgs(
        limit=ns.limit,
        video_limit=ns.video_limit,
    )

    chroma = ChromaArgs(
        host=ns.chroma_host,
        port=ns.chroma_port,
        use_ssl=ns.chroma_use_ssl,
        access_key=ns.chroma_access_key,
        secret_key=ns.chroma_secret_key,
        bearer_token=ns.chroma_bearer_token,
        timeout=ns.chroma_timeout,
    )

    return AllArgs(minio=minio, landing=landing,temporal = temporal, chroma=chroma)


def connect_minio(args_minio):
    """
    Build and return an S3-compatible client for MinIO.

    This function serves as the connection initializer for all subsequent
    data lake operations. It parses the connection parameters (endpoint,
    access key, secret key, and logging level) from the command line and
    creates a boto3 S3 client instance configured for MinIO compatibility.

    """
    logging.basicConfig(
        level=getattr(logging, args_minio.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(name)s: %(message)s"
    )

    s3 = make_s3_client(args_minio.endpoint, args_minio.access_key, args_minio.secret_key)
    return s3

def landing_init(client, args_landing):
    """
    This function prepares the basic folder-like structure used for raw
    and temporary data ingestion in a data lake. It uses an existing
    S3-compatible client to:
        - Create the target bucket if it does not already exist.
        - Ensure that all required folder prefixes (e.g., temporal/persistent)
          are materialized within the bucket.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).

    """

    ensure_bucket(client, args_landing.bucket)
    ensure_prefixes(client, args_landing.bucket, args_landing.prefixes)


def temporal_landing_init(client,args):
    """
    Initialize the temporal landing zone by loading datasets and storing them
    into the 'landing-zone/temporal-landing/' folder on MinIO.

    The function connects to the S3-compatible MinIO service, retrieves
    datasets from predefined sources (e.g., Hugging Face), and uploads them
    to the temporal landing area for further processing.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).

    """
    logger = logging.getLogger(__name__)
    limit = args.limit
    split = "train"
    ds1_root ="FronkonGames/steam-games-dataset"
    ds2_root ="atalaydenknalbant/rawg-games-dataset"
    # -------- Temporal landing datasets --------

    # load data
    ds1_raw = load_huggingface_dataset(ds1_root, split,None)
    ds2_raw = load_huggingface_dataset(ds2_root, split, None)


    # We are going to use the first n obs from each dataset for testing purposes
    if limit == -1:
        ds1 = ds1_raw
        ds2 = ds2_raw
    else :
        ds1 = ds1_raw[0:limit]
        ds2 = ds2_raw[0:limit]
    # Print the number of rows of each subdataset
    logger.info("Prepared subsets: ds1=%d rows, ds2=%d rows", len(ds1['About the game']), len(ds2['description']))


    # We are interested on Text, Image and Video data
    # We can find each of these data in the following columns
    # ds1: "About the game" (Text), "Header image" (Image), "Screenshots" (Image), "Movies" (Video)
    # ds2: "description" (Text), "background_image" (Image), "background_image_additional" (Image), "short_screenshots" (Image)
    # By combing both datasets, we assume there will be duplicates of games
    # Uploading text files (combining both datasets)
    upload_strings_separately("landing-zone", client, strings =
                              ds1['About the game'] +
                              ds2['description'],
                              path = "temporal-landing/")

    # Uploading image files (combining both datasets)
    upload_media_from_links("landing-zone", client, links=
    ds1['Header image'] + ds2['background_image'],  #ds1['Header image'] + ds2['background_image_additional']
                            path="temporal-landing/")  # If this process is taking too long, we can just skip the screeshots

    # Uploading video files
    upload_media_from_links("landing-zone", client, links=
    ds1['Movies'],  # It's recommended to upload only a few videos due to MinioIO's storage size
                            path="temporal-landing/", prefix="video", limit=args.video_limit)  # Here we are only selecting the first 10 videos

def persistent_landing_init(client):
    """
    Initialize the *persistent* landing zone layout in S3/MinIO.

    This function ensures that a stable, long-lived landing area exists for
    production data (as opposed to the short-lived “temporal” zone). It will:
      - creates (or verifies) the destination bucket,
      - materializes the canonical prefix tree under `landing-zone/persistent/`
         (e.g., `texts/`, `images/`, `videos/`, etc.),
      - classifies incoming objects by Content-Type (text/*, image/*, video/*)
         and moves/copies them into the corresponding subfolders.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    """

    ensure_prefixes(client, "landing-zone", ["persistent-landing/texts", "persistent-landing/images", "persistent-landing/videos"])
    move_files(client, "landing-zone")

def formatted_init(client):
    """
    This function replicates all objects from the persistent landing area
    (i.e., 'landing-zone/persistent-landing/') into a new S3 bucket named
    'formatted-zone'. The target bucket will be created automatically if
    it does not exist.

    The formatted zone acts as the next stage of the data lake pipeline,
    where raw data from the landing zone is stored in a consistent,
    structured format ready for downstream processing or analytics.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    """
    replicate_bucket(client, "landing-zone", "formatted-zone", "persistent-landing")

def formatted(client):
    """
    Run the full formatting pipeline on the 'formatted-zone' bucket.

    This function processes all supported media types stored under the
    `formatted-zone` structure in S3/MinIO. Specifically, it performs:

      - Image normalization — converts all non-PNG images to `.png`
        under `formatted-zone/images/`.
      - Document normalization — converts all supported document formats
        (e.g., PDF, DOCX, HTML, RTF, ODT) to plain-text `.txt` with utf-8 encoding
        under `formatted-zone/texts/`.
      - Video normalization — converts all non-MP4 videos to `.mp4`
        under `formatted-zone/videos/`.

    The goal is to ensure all media in the formatted zone share consistent,
    standardized formats for downstream processing.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    """
    convert_images_to_png(client, "formatted-zone", "images/")
    convert_documents_to_txt(client, "formatted-zone", "texts/")
    convert_videos_to_mp4(client, "formatted-zone", "videos/")

def trusted_init(client):
    """
    This function replicates all objects from the formatted data area
    (i.e., 'formatted-zone/') into a new S3 bucket named 'trusted-zone'.
    The target bucket will be created automatically if it does not exist.

    The trusted zone represents the final curated layer in the data lake
    pipeline, where standardized and validated data from the formatted zone
    is consolidated for reliable consumption by downstream systems,
    analytics workflows, or machine learning pipelines.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    """
    replicate_bucket(client, "formatted-zone", "trusted-zone")

def trusted(client):
    """
    The function executes the end-to-end data hygiene/normalization flows for the bucket.

    Each sub-pipeline is responsible for generating its own quality report,
    removing low-quality/duplicate assets, normalizing content (e.g., RGB resize
    for images, decoding/probing for videos, cleaning/translation/chunking for
    text), and emitting progress/summary logs.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    """

    preprocess_image(client, "trusted-zone","images/")
    preprocess_video(client, "trusted-zone","videos/")
    clean_text(client, "trusted-zone","texts/")

def connect_chromaDB(args_chroma):
    """
    Initialize and validate a connection to a ChromaDB server.

    This function constructs a ChromaDB client using the provided connection
    parameters (host, port, authentication, and timeout). It also validates
    the port number before attempting to connect.
    """

    # Validate port range before creating the client
    if not (0 < args_chroma.port < 65536):
        raise ValueError(f"Invalid port: {args_chroma.port}")

    # Construct ChromaDB client using provided connection parameters
    client = make_chromaDB_client(
        host=args_chroma.host,
        port=args_chroma.port,
        use_ssl=args_chroma.use_ssl,
        access_key=args_chroma.access_key,
        secret_key=args_chroma.secret_key,
        bearer_token=args_chroma.bearer_token,
        timeout=args_chroma.timeout,
    )

    return client

def exploitation(chroma_client, s3_client):
    image_collection = create_chroma_collection(chroma_client, "images")
    image_model, image_preprocess,image_device = get_image_model()
    images_to_embeddings(s3_client,"trusted-zone",image_collection,image_preprocess,image_model,image_device,"images/")

    video_collection = create_chroma_collection(chroma_client,"videos")
    video_model, video_processors,video_device = get_video_model()
    videos_to_embeddings(s3_client,"trusted-zone","videos/",video_collection,video_processors,video_model,video_device)

    text_collection = create_chroma_collection(chroma_client, "texts")
    tokenizer, text_model,  text_device = get_text_model()
    texts_to_embeddings(s3_client,"trusted-zone",text_collection,tokenizer,text_model,text_device,"texts/")

    image_text_collection = create_chroma_collection(chroma_client, "texts_images")
    image_collection = create_chroma_collection(chroma_client, "images")
    combining_image_text(image_collection,text_collection,image_text_collection)



def multi_modal_task_execution(chroma_client, s3_client):

    example_text = get_text(s3_client,"trusted-zone", "texts/text_1761318441201.txt")
    text_collection = create_chroma_collection(chroma_client, "texts")
    result1 = get_similar_text(s3_client,text_collection,example_text)
    text_example_2 = "Undertale is a unique role-playing game where players navigate a world filled with quirky monsters. Choices matter: you can fight, flee, or befriend enemies, affecting the story and multiple endings. The game combines retro-style graphics, witty humor, and emotional storytelling, offering a deep, player-driven experience that challenges traditional RPG mechanics."
    result2 = get_similar_text(s3_client,text_collection,text_example_2)

    image_collection = create_chroma_collection(chroma_client, "images")
    example_image = get_image(s3_client,"trusted-zone","images/image_1761318414668.png")
    result3 = get_similar_image(s3_client,image_collection,example_image)

    video_collection = create_chroma_collection(chroma_client, "videos")
    example_video = get_video(s3_client, "trusted-zone", "videos/video_1761318468194.mp4")
    #result3 = get_similar_video(s3_client, video_collection, example_video)

    image_text_collection = create_chroma_collection(chroma_client, "texts_images")
    result4 = find_k_similars_by_file(s3_client,image_text_collection,example_text,"text")

    image_text_collection = create_chroma_collection(chroma_client, "texts_images")

    example_image = get_image(s3_client,"trusted-zone","images/image_1761318414668.png")

    result5 = get_recommendation(s3_client,image_text_collection,"Please recommend me an open-world action RPG game with exploration similar to the one shown in the image.",example_image)



if __name__ == "__main__":
    args = parse_all_args()
    s3_client = connect_minio(args.minio)
    landing_init(s3_client,args.landing)
    temporal_landing_init(s3_client,args.temporal)
    persistent_landing_init(s3_client)
    formatted_init(s3_client)
    formatted(s3_client)
    trusted_init(s3_client)
    trusted(s3_client)
    chroma_client = connect_chromaDB(args.chroma)

    exploitation(chroma_client,s3_client)
    multi_modal_task_execution(chroma_client,s3_client)