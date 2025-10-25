import argparse
import logging

from Landing_Zone.landing_zone import make_s3_client,ensure_bucket,ensure_prefixes
from Landing_Zone.temporal_landing import load_huggingface_dataset, upload_strings_separately,upload_media_from_links
from Landing_Zone.persistent_landing import move_files
from utils.utils import replicate_bucket
from Formatted_Zone.formatted_zone_homogenizer_images import convert_images_to_png
from Formatted_Zone.formatted_zone_homogenizer_texts import convert_documents_to_txt

def parse_args_credentials_init():
    """
    Parse command-line arguments for connect minio.
    """
    parser = argparse.ArgumentParser(description="connect to minio server.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9000",
                        help="MinIO/S3 endpoint URL.")
    parser.add_argument("--access-key", default="minioadmin",
                        help="Access key (username) for MinIO/S3.")
    parser.add_argument("--secret-key", default="minioadmin",
                        help="Secret key (password) for MinIO/S3.")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args()


def parse_args_landing_init():
    """
    Parse command-line arguments for create bucket of landing zone.
    """
    parser = argparse.ArgumentParser(description="Initialize MinIO buckets and folder-like prefixes.")
    parser.add_argument("--bucket", default="landing-zone",
                        help="Bucket name to ensure/create.")
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=["temporal-landing/", "persistent-landing/"],
        help="Folder-like prefixes to ensure under the bucket (trailing slash is optional).",
    )

    return parser.parse_args()


def parse_args_temporal_landing():
    """
    Parse command-line arguments for temporal landing (dataset root etc...).
    """
    parser = argparse.ArgumentParser(
        description="Load two Hugging Face datasets and print row counts (non-streaming)."
    )
    parser.add_argument("--ds1", default="FronkonGames/steam-games-dataset",
                        help="First dataset identifier on Hugging Face Hub.")
    parser.add_argument("--ds2", default="atalaydenknalbant/rawg-games-dataset",
                        help="Second dataset identifier on Hugging Face Hub.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to load (e.g., 'train', 'test', 'validation').")
    parser.add_argument("--cache-dir", default=None,
                        help="HF datasets cache directory.")
    return parser.parse_args()
    args = parse_args_landing_init()

def connect_minio():
    """
    Build and return an S3-compatible client for MinIO.

    This function serves as the connection initializer for all subsequent
    data lake operations. It parses the connection parameters (endpoint,
    access key, secret key, and logging level) from the command line and
    creates a boto3 S3 client instance configured for MinIO compatibility.

    """
    args = parse_args_credentials_init()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(name)s: %(message)s"
    )

    s3 = make_s3_client(args.endpoint, args.access_key, args.secret_key)
    return s3

def landing_init(client):
    """
    This function prepares the basic folder-like structure used for raw
    and temporary data ingestion in a data lake. It uses an existing
    S3-compatible client to:
        - Create the target bucket if it does not already exist.
        - Ensure that all required folder prefixes (e.g., temporal/persistent)
          are materialized within the bucket.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).

    """
    args = parse_args_landing_init()

    ensure_bucket(client, args.bucket)
    ensure_prefixes(client, args.bucket, args.prefixes)


def temporal_landing_init(client, limit: int = 500):
    """
    Initialize the temporal landing zone by loading datasets and storing them
    into the 'landing-zone/temporal-landing/' folder on MinIO.

    The function connects to the S3-compatible MinIO service, retrieves
    datasets from predefined sources (e.g., Hugging Face), and uploads them
    to the temporal landing area for further processing.

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).

    """
    logger = logging.getLogger(__name__)
    args = parse_args_temporal_landing()


    # load data
    ds1_raw = load_huggingface_dataset(args.ds1, args.split, args.cache_dir)
    ds2_raw = load_huggingface_dataset(args.ds2, args.split, args.cache_dir)


    # We are going to use the first n obs from each dataset for testing purposes
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
                              path = "temporal-landing/", limit= 100)

    # Uploading image files (combining both datasets)
    upload_media_from_links("landing-zone", client, links=
    ds1['Header image'] + ds2['background_image'],  #ds1['Header image'] + ds2['background_image_additional']
                            path="temporal-landing/", limit= 100)  # If this process is taking too long, we can just skip the screeshots

    # Uploading video files
    upload_media_from_links("landing-zone", client, links=
    ds1['Movies'],  # It's recommended to upload only a few videos due to MinioIO's storage size
                            path="temporal-landing/", prefix="video", limit=10)  # Here we are only selecting the first 10 videos

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

    client          : obj                   - S3-compatible client (e.g., boto3.client("s3")).
    """
    convert_images_to_png(client, "formatted-zone", "images/")
    convert_documents_to_txt(client, "formatted-zone", "texts/")





if __name__ == "__main__":
    s3_client = connect_minio()
    #landing_init(s3_client)
    #temporal_landing_init(s3_client)
    #persistent_landing_init(s3_client)
    #formatted_init(s3_client)
    #formatted(s3_client)
