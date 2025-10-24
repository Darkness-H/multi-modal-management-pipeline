import argparse
from Landing_Zone.landing_zone import *
from Landing_Zone.temporal_landing import *
from Landing_Zone.persistent_landing import move_files
from utils.utils import *
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
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).")
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
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level, e.g., DEBUG, INFO, WARNING, ERROR.")
    return parser.parse_args()
    args = parse_args_landing_init()

def connect_minio():
    """
    build client
    """
    args = parse_args_credentials_init()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )

    client = make_s3_client(args.endpoint, args.access_key, args.secret_key)
    return client

def landing_init(client):
    """
    Ensure bucket and required prefixes exist.
    """
    args = parse_args_landing_init()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )

    ensure_bucket(client, args.bucket)
    ensure_prefixes(client, args.bucket, args.prefixes)


def temporal_landing_init(client, limit: int = 500):
    """
    Initialize the temporal landing zone by loading datasets and storing them
    into the 'landing-zone/temporal-landing/' folder on MinIO.

    The function connects to the S3-compatible MinIO service, retrieves
    datasets from predefined sources (e.g., Hugging Face), and uploads them
    to the temporal landing area for further processing.
    """
    args = parse_args_temporal_landing()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    logging.info("Loading dataset 1: %s (split=%s)", args.ds1, args.split)
    ds1_raw = load_huggingface_dataset(args.ds1, args.split, args.cache_dir)
    print(f"The dataset 1 ('{args.ds1}'/{args.split}) contains {ds1_raw.num_rows} rows")

    logging.info("Loading dataset 2: %s (split=%s)", args.ds2, args.split)
    ds2_raw = load_huggingface_dataset(args.ds2, args.split, args.cache_dir)
    print(f"The dataset 2 ('{args.ds2}'/{args.split}) contains {ds2_raw.num_rows} rows")
    # We are going to use the first n obs from each dataset for testing purposes
    ds1 = ds1_raw[0:limit]
    ds2 = ds2_raw[0:limit]
    # Print the number of rows of each subdataset
    print(f"The subdataset 1 contains {len(ds1['About the game'])} rows")
    print(f"The subdataset 2 contains {len(ds2['description'])} rows")


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
    """
    ensure_prefixes(client, "landing-zone", ["persistent-landing/texts", "persistent-landing/images", "persistent-landing/videos"])
    move_files(client, "landing-zone")

def formatted_init(client):
    """

    """
    replicate_bucket(client, "landing-zone", "formatted-zone", "persistent-landing")



if __name__ == "__main__":
    client = connect_minio()
    #landing_init(client)
    #temporal_landing_init(client)
    #persistent_landing_init(client)
    formatted_init(client)
