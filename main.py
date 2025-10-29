import argparse
import logging
import os
import queue
import sys
import threading
import warnings
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, List

from pathlib import Path

from src.utils.file_utils import prepare_file_for_search
from adodbapi import connect

from src.utils.bucket_utils import replicate_bucket, ensure_bucket, ensure_prefixes, get_random_s3_key
from src.utils.make_conecction import make_s3_client, make_chromaDB_client
from src.utils.collection_utils import create_chroma_collection

from src.Landing_Zone.temporal_landing import load_huggingface_dataset, upload_strings_separately,upload_media_from_links
from src.Landing_Zone.persistent_landing import move_files

from src.Formatted_Zone.formatted_zone_homogenizer_images import convert_images_to_png
from src.Formatted_Zone.formatted_zone_homogenizer_texts import convert_documents_to_txt
from src.Formatted_Zone.formatted_zone_homogenizer_videos import convert_videos_to_mp4

from src.Trusted_Zone.trusted_zone_image_quality_processes import preprocess_image
from src.Trusted_Zone.trusted_zone_video_quality_process import preprocess_video
from src.Trusted_Zone.trusted_zone_text_quality_processes import clean_text

from src.exploitation_zone.exploitation_zone_image_embeddings import images_to_embeddings, get_image_model
from src.exploitation_zone.exploitation_zone_text_embeddings import get_text_model, texts_to_embeddings
from src.exploitation_zone.utils_exploitation.getter import get_text, get_image, get_video
from src.exploitation_zone.exploitation_zone_video_embeddings import videos_to_embeddings, get_video_model
from src.exploitation_zone.embeddings_combination import combining_image_text

from src.multi_modal_task.same_modal_similarity_search import get_similar_text, get_similar_image, get_similar_video
from src.multi_modal_task.multi_modal_similarity_search import find_k_similars_by_file
from src.multi_modal_task.generative_recommendation import get_recommendation

import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from tkinter import ttk

# ---- Dataclasses to carry structured args ----
@dataclass
class InitializationArgs:
    endpoint: str
    access_key: str
    secret_key: str
    log_level: str


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
    ini: InitializationArgs
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
    g_ini = parser.add_argument_group("MinIO / S3")
    g_ini.add_argument("--minio-endpoint", default="http://127.0.0.1:9000",
                         help="MinIO/S3 endpoint URL.")
    g_ini.add_argument("--minio-access-key", default="minioadmin",
                         help="Access key (username) for MinIO/S3.")
    g_ini.add_argument("--minio-secret-key", default="minioadmin",
                         help="Secret key (password) for MinIO/S3.")
    g_ini.add_argument("--minio-log-level", default="INFO",
                         help="Logging level: DEBUG|INFO|WARNING|ERROR.")




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

    minio = InitializationArgs(
        endpoint=ns.minio_endpoint,
        access_key=ns.minio_access_key,
        secret_key=ns.minio_secret_key,
        log_level=ns.minio_log_level,
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

    return AllArgs(ini=minio, temporal = temporal, chroma=chroma)


def connect_minio(args_minio):
    """
    Build and return an S3-compatible client for MinIO.

    This function serves as the connection initializer for all subsequent
    data lake operations. It parses the connection parameters (endpoint,
    access key, secret key, and logging level) from the command line and
    creates a boto3 S3 client instance configured for MinIO compatibility.

    """
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    os.makedirs("logs", exist_ok=True)
    fh = TimedRotatingFileHandler(
        filename=os.path.join("logs", "app.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logging.basicConfig(
        level=getattr(logging, args_minio.log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[
        logging.StreamHandler(sys.stdout),
        fh,
        ],
        force=True
    )

    logger = logging.getLogger(__name__)

    logger.info("Logging initialized")
    s3 = make_s3_client(args_minio.endpoint, args_minio.access_key, args_minio.secret_key)
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

    ensure_bucket(client,"landing-zone")
    ensure_prefixes(client, "landing-zone", ["temporal-landing/", "persistent-landing/"])


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
    text_key = get_random_s3_key(s3_client,"trusted-zone","texts/")
    example_text = get_text(s3_client,"trusted-zone", text_key)
    print("find similar text for :\n" + example_text)
    text_collection = create_chroma_collection(chroma_client, "texts")
    get_similar_text(s3_client,text_collection,example_text)
    text_example_2 = "Undertale is a unique role-playing game where players navigate a world filled with quirky monsters. Choices matter: you can fight, flee, or befriend enemies, affecting the story and multiple endings. The game combines retro-style graphics, witty humor, and emotional storytelling, offering a deep, player-driven experience that challenges traditional RPG mechanics."
    print("find similar text for :\n" + text_example_2)
    get_similar_text(s3_client,text_collection,text_example_2)

    image_collection = create_chroma_collection(chroma_client, "images")
    image_key = get_random_s3_key(s3_client, "trusted-zone", "images/")
    example_image = get_image(s3_client,"trusted-zone",image_key)
    get_similar_image(s3_client,image_collection,example_image)

    video_collection = create_chroma_collection(chroma_client, "videos")
    video_key = get_random_s3_key(s3_client, "trusted-zone", "videos/")
    example_video = get_video(s3_client, "trusted-zone", video_key)
    get_similar_video(s3_client, video_collection, example_video)

    image_text_collection = create_chroma_collection(chroma_client, "texts_images")
    find_k_similars_by_file(s3_client,image_text_collection,example_text,"text")

    image_text_collection = create_chroma_collection(chroma_client, "texts_images")
    result5 = get_recommendation(s3_client,image_text_collection,"Please recommend me an open-world action RPG game with exploration similar to the one shown in the image.",example_image)

#user interface definitions
class GUI(tk.Tk):
    def __init__(self,defaults: AllArgs):
        super().__init__()
        self.defaults = defaults
        self.title("Game Recommendation System")
        self.geometry("1080x900")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self._build_home_tab(nb)
        self._build_similarity_tab(nb)
        self._build_multi_modal_similarity_tab(nb)
        self._build_generative_tab(nb)
        self._build_temporal_tab(nb, defaults.temporal)
        self.s3_client = connect_minio(defaults.ini)
        self.chroma_client = connect_chromaDB(defaults.chroma)

        frm = ttk.Frame(self)
        frm.pack(fill="x", padx=10, pady=6)

    def _goto_tab_by_title(self, nb, title: str):
        for tab_id in nb.tabs():
            if nb.tab(tab_id, "text") == title:
                nb.select(tab_id)
                return

    def _build_home_tab(self,nb):
        f = ttk.Frame(nb); nb.add(f, text="Home")
        center = ttk.Frame(f)
        center.pack(expand=True)

        ttk.Button(center,width=28, text="Run pipeline", command=self._on_run).grid(row=0, column=0, padx=8, pady=6)
        ttk.Button(center,width=28, text="Similarity search",  command=lambda: self._goto_tab_by_title(nb, "Similarity")).grid(row=1, column=0, padx=8, pady=6)
        ttk.Button(center,width=28, text="Multi-modal similarity search", command=lambda: self._goto_tab_by_title(nb, "Multi-modal")).grid(row=2, column=0, padx=8, pady=6)
        ttk.Button(center,width=28, text="Generative recommendation",  command=lambda: self._goto_tab_by_title(nb, "Recommendation")).grid(row=3, column=0, padx=8, pady=6)
        ttk.Button(center,width=28, text="Close", command=self.destroy).grid(row=4, column=0, padx=8, pady=6)


    def _build_similarity_tab(self,nb):
        f = ttk.Frame(nb); nb.add(f, text="Similarity")
        style = ttk.Style()
        style.configure("Equal.TButton", padding=(16, 10))
        BTN_WIDTH = 14
        for c in range(3):
            f.grid_columnconfigure(c, weight=(1 if c == 1 else 0))
        pad = dict(padx=8, pady=6)

        if not hasattr(self, "var_file_type"):
            self.var_file_type = tk.StringVar(value="Queries")
        if not hasattr(self, "var_file_path"):
            self.var_file_path = tk.StringVar()
        if not hasattr(self, "var_input_mode"):
            self.var_input_mode = tk.StringVar(value="queries")

        ttk.Label(f, text="File type").grid(row=0, column=0, sticky="e", **pad)
        cb = ttk.Combobox(
            f,
            textvariable=self.var_file_type,
            values=["Queries", "Text", "Image"],
            state="readonly",
            width=20
        )
        cb.grid(row=0, column=1, sticky="w", **pad)
        queries_frame = ttk.Frame(f)
        queries_frame.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        queries_frame.grid_columnconfigure(0, weight=0)
        queries_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(queries_frame, text="Queries").grid(row=0, column=0, sticky="ne", padx=(0, 8))
        self.txt_queries = tk.Text(queries_frame, height=6, width=50)
        self.txt_queries.grid(row=0, column=1, sticky="ew")

        file_frame = ttk.Frame(f)
        file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        file_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(file_frame, text="File").grid(row=0, column=0, sticky="e", padx=(0, 8))
        ent_file = ttk.Entry(file_frame, textvariable=self.var_file_path)
        ent_file.grid(row=0, column=1, sticky="ew")
        btn_browse = ttk.Button(
            file_frame, text="Browse…",
            command=lambda: self._browse_file_with_type()
        )
        btn_browse.grid(row=0, column=2, sticky="w", padx=(8, 0))

        center = ttk.Frame(f)
        center.grid(row=4, column=0, columnspan=3, pady=(8, 12))
        center.grid_columnconfigure(0, weight=1)



        ttk.Button(
            center, text="Search", width=BTN_WIDTH, style="Equal.TButton",
            command=lambda: self._on_run_sim(ent_file.get(),cb.get())
        ).grid(row=0, column=0, padx=8, pady=0, sticky="")

        self.reply = ttk.Frame(f, padding=8)
        self.reply.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.reply.grid_columnconfigure(0, weight=0)
        self.reply.grid_columnconfigure(1, weight=1)
        ttk.Label(self.reply, text="Reply").grid(row=0, column=0, sticky="ne", padx=(0, 8))



        ttk.Button(
            center, text="Home", width=BTN_WIDTH, style="Equal.TButton",
            command=lambda: self._goto_tab_by_title(nb, "Home")
        ).grid(row=1, column=0, padx=8, pady=0, sticky="")

        ttk.Button(
            center, text="Close",width=BTN_WIDTH, style="Equal.TButton",
            command=self.destroy
        ).grid(row=2, column=0, sticky="e", padx=8, pady=8)
        def _on_type_change(_=None):
            t = (self.var_file_type.get() or "").lower()

            if t == "queries":

                queries_frame.grid()
                file_frame.grid_remove()
            else:
                file_frame.grid()
                queries_frame.grid_remove()

        cb.bind("<<ComboboxSelected>>", _on_type_change)
        _on_type_change()

    def _build_multi_modal_similarity_tab(self,nb):
        f = ttk.Frame(nb); nb.add(f, text="Multi-modal")
        style = ttk.Style()
        style.configure("Equal.TButton", padding=(16, 10))
        BTN_WIDTH = 14
        for c in range(3):
            f.grid_columnconfigure(c, weight=(1 if c == 1 else 0))
        pad = dict(padx=8, pady=6)

        if not hasattr(self, "var_file_type"):
            self.var_file_type = tk.StringVar(value="Queries")
        if not hasattr(self, "var_file_path"):
            self.var_file_path = tk.StringVar()
        if not hasattr(self, "var_input_mode"):
            self.var_input_mode = tk.StringVar(value="queries")

        ttk.Label(f, text="File type").grid(row=0, column=0, sticky="e", **pad)
        cb = ttk.Combobox(
            f,
            textvariable=self.var_file_type,
            values=["Queries", "Text", "Image"],
            state="readonly",
            width=20
        )
        cb.grid(row=0, column=1, sticky="w", **pad)
        queries_frame = ttk.Frame(f)
        queries_frame.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        queries_frame.grid_columnconfigure(0, weight=0)
        queries_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(queries_frame, text="Queries").grid(row=0, column=0, sticky="ne", padx=(0, 8))
        self.txt_queries1 = tk.Text(queries_frame, height=6, width=50)
        self.txt_queries1.grid(row=0, column=1, sticky="ew")

        file_frame = ttk.Frame(f)
        file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        file_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(file_frame, text="File").grid(row=0, column=0, sticky="e", padx=(0, 8))
        ent_file = ttk.Entry(file_frame, textvariable=self.var_file_path)
        ent_file.grid(row=0, column=1, sticky="ew")
        btn_browse = ttk.Button(
            file_frame, text="Browse…",
            command=lambda: self._browse_file_with_type()
        )
        btn_browse.grid(row=0, column=2, sticky="w", padx=(8, 0))

        center = ttk.Frame(f)
        center.grid(row=4, column=0, columnspan=3, pady=(8, 12))
        center.grid_columnconfigure(0, weight=1)



        ttk.Button(
            center, text="Search", width=BTN_WIDTH, style="Equal.TButton",
            command=lambda: self._on_run_multi_sim(ent_file.get(),cb.get())
        ).grid(row=0, column=0, padx=8, pady=0, sticky="")

        self.reply1 = ttk.Frame(f, padding=8)
        self.reply1.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.reply1.grid_columnconfigure(0, weight=0)
        self.reply1.grid_columnconfigure(1, weight=1)
        ttk.Label(self.reply1, text="Reply").grid(row=0, column=0, sticky="ne", padx=(0, 8))



        ttk.Button(
            center, text="Home", width=BTN_WIDTH, style="Equal.TButton",
            command=lambda: self._goto_tab_by_title(nb, "Home")
        ).grid(row=1, column=0, padx=8, pady=0, sticky="")

        ttk.Button(
            center, text="Close",width=BTN_WIDTH, style="Equal.TButton",
            command=self.destroy
        ).grid(row=2, column=0, sticky="e", padx=8, pady=8)
        def _on_type_change(_=None):
            t = (self.var_file_type.get() or "").lower()

            if t == "queries":

                queries_frame.grid()
                file_frame.grid_remove()
            else:
                file_frame.grid()
                queries_frame.grid_remove()

        cb.bind("<<ComboboxSelected>>", _on_type_change)
        _on_type_change()

    def _build_generative_tab(self,nb):
        f = ttk.Frame(nb); nb.add(f, text="Recommendation")
        style = ttk.Style()
        style.configure("Equal.TButton", padding=(16, 10))
        BTN_WIDTH = 14
        for c in range(3):
            f.grid_columnconfigure(c, weight=(1 if c == 1 else 0))
        pad = dict(padx=8, pady=6)

        if not hasattr(self, "var_file_type"):
            self.var_file_type = tk.StringVar(value="Queries")
        if not hasattr(self, "var_file_path"):
            self.var_file_path = tk.StringVar()
        if not hasattr(self, "var_input_mode"):
            self.var_input_mode = tk.StringVar(value="queries")

        ttk.Label(f, text="File type").grid(row=0, column=0, sticky="e", **pad)
        cb = ttk.Combobox(
            f,
            textvariable=self.var_file_type,
            values=["Queries", "Text"],
            state="readonly",
            width=20
        )
        cb.grid(row=0, column=1, sticky="w", **pad)
        queries_frame = ttk.Frame(f)
        queries_frame.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        queries_frame.grid_columnconfigure(0, weight=0)
        queries_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(queries_frame, text="Queries").grid(row=0, column=0, sticky="ne", padx=(0, 8))
        self.txt_queries2 = tk.Text(queries_frame, height=6, width=50)
        self.txt_queries2.grid(row=0, column=1, sticky="ew")

        file_frame = ttk.Frame(f)
        file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        file_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Text").grid(row=0, column=0, sticky="e", padx=(0, 8))
        ent_file = ttk.Entry(file_frame, textvariable=self.var_file_path)
        ent_file.grid(row=0, column=1, sticky="ew")
        btn_browse = ttk.Button(
            file_frame, text="Browse…",
            command=lambda: self._browse_file_with_type()
        )
        btn_browse.grid(row=0, column=2, sticky="w", padx=(8, 0))

        file_frame2 = ttk.Frame(f)
        file_frame2.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        file_frame2.grid_columnconfigure(1, weight=1)

        ttk.Label(file_frame2, text="Image").grid(row=0, column=0, sticky="e", padx=(0, 8))
        ent_file2 = ttk.Entry(file_frame2, textvariable=self.var_file_path)
        ent_file2.grid(row=0, column=1, sticky="ew")
        btn_browse2 = ttk.Button(
            file_frame2, text="Browse…",
            command=lambda: self._browse_file_with_type()
        )
        btn_browse2.grid(row=0, column=2, sticky="w", padx=(8, 0))

        center = ttk.Frame(f)
        center.grid(row=5, column=0, columnspan=3, pady=(8, 12))
        center.grid_columnconfigure(0, weight=1)

        ttk.Button(
            center, text="Generate", width=BTN_WIDTH, style="Equal.TButton",
            command=lambda: self._on_run_gen(ent_file.get(),ent_file2.get(),cb.get())
        ).grid(row=0, column=0, padx=8, pady=0, sticky="")

        self.reply2 = ttk.Frame(f, padding=8)
        self.reply2.grid(row=4, column=0, columnspan=3, sticky="nsew")
        self.reply2.grid_columnconfigure(0, weight=0)
        self.reply2.grid_columnconfigure(1, weight=1)
        ttk.Label(self.reply2, text="Reply").grid(row=0, column=0, sticky="ne", padx=(0, 8))

        ttk.Button(
            center, text="Home", width=BTN_WIDTH, style="Equal.TButton",
            command=lambda: self._goto_tab_by_title(nb, "Home")
        ).grid(row=1, column=0, padx=8, pady=0, sticky="")

        ttk.Button(
            center, text="Close",width=BTN_WIDTH, style="Equal.TButton",
            command=self.destroy
        ).grid(row=2, column=0, sticky="e", padx=8, pady=8)
        def _on_type_change(_=None):
            t = (self.var_file_type.get() or "").lower()
            if t == "queries":

                queries_frame.grid()
                file_frame.grid_remove()
            else:
                file_frame.grid()
                queries_frame.grid_remove()

        cb.bind("<<ComboboxSelected>>", _on_type_change)
        _on_type_change()

    def _build_temporal_tab(self, nb, temporal: TemporalArgs):
        f = ttk.Frame(nb);
        nb.add(f, text="Temporal landing settings")
        pad = dict(padx=8, pady=6)
        self.var_limit = tk.IntVar(value=int(temporal.limit))
        self.var_vlimit = tk.IntVar(value=int(temporal.video_limit))

        ttk.Label(f, text="Limit").grid(row=0, column=0, sticky="e", **pad)
        ttk.Spinbox(f, from_=-1, to=100000, textvariable=self.var_limit, width=10).grid(row=0, column=1, sticky="w",
                                                                                        **pad)
        ttk.Label(f, text="Video Limit").grid(row=1, column=0, sticky="e", **pad)
        ttk.Spinbox(f, from_=0, to=10000, textvariable=self.var_vlimit, width=10).grid(row=1, column=1, sticky="w", **pad)

    def collect_config(self) -> AllArgs:

        temporal = TemporalArgs(
            limit=int(self.var_limit.get()),
            video_limit=int(self.var_vlimit.get()),
        )


        return AllArgs(ini=None,  temporal=temporal, chroma=None)

    def _on_run(self):
        try:
            cfg = self.collect_config()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return
        t = threading.Thread(target=self._run_pipeline, args=(cfg,), daemon=True)
        t.start()

    def _run_pipeline(self, cfg):
        try:
            landing_init(self.s3_client)
            temporal_landing_init(self.s3_client,cfg.temporal)
            persistent_landing_init(self.s3_client)
            formatted_init(self.s3_client)
            formatted(self.s3_client)
            trusted_init(self.s3_client)
            trusted(self.s3_client)
            exploitation(self.chroma_client,self.s3_client)
        except Exception as e:
            print(e)

    def _on_run_sim(self,file,file_type):
        file_type = file_type.lower()
        if file_type == "queries":
            file = self.txt_queries.get("1.0", "end-1c").strip()
        else:
            file = prepare_file_for_search(file,file_type)
        t = threading.Thread(target=self._run_similarity_search, args=(file,file_type), daemon=True)
        t.start()

    def _run_similarity_search(self, file,file_type: str):
        res = None


        self.after(0, print(file_type))
        if file_type == "text":
            collection = create_chroma_collection(self.chroma_client,"texts")
            res = get_similar_text(self.s3_client, collection, file)
            self.after(0, lambda: self._handle_text_result(res,file_type))

        elif file_type == "image":
            collection = create_chroma_collection(self.chroma_client,"images")
            res = get_similar_image(self.s3_client, collection, file)
            self.after(0, lambda: self._handle_text_result(res,file_type))


        elif file_type == "queries":
            collection = create_chroma_collection(self.chroma_client,"texts")
            res = get_similar_text(self.s3_client, collection, file)
            self.after(0, lambda: self._handle_text_result(res,file_type))

        return res

    def _on_run_multi_sim(self,file,file_type):
        file_type = file_type.lower()
        if file_type == "queries":
            file = self.txt_queries1.get("1.0", "end-1c").strip()
            file_type = "text"
        else:
            file = prepare_file_for_search(file,file_type)
        t = threading.Thread(target=self._run_multi_modal_similarity_search, args=(file,file_type), daemon=True)
        t.start()

    def _run_multi_modal_similarity_search(self, file,file_type):
        image_text_collection = create_chroma_collection(self.chroma_client, "texts_images")
        res1,res2=find_k_similars_by_file(self.s3_client, image_text_collection, file, file_type)
        self.after(0,lambda: self._handle_multi_modal_result(res1,res2))

    def _on_run_gen(self,text_file,img_file,file_type):
        file_type = file_type.lower()
        text = None
        img = None
        if file_type == "queries":
            text = self.txt_queries1.get("1.0", "end-1c").strip()
        elif text_file:
            text = prepare_file_for_search(text_file,file_type)

        if img_file:
            img = prepare_file_for_search(img_file,"image")

        t = threading.Thread(target=self._run_gen, args=(text,img), daemon=True)
        t.start()

    def _run_gen(self,text, image):
        image_text_collection = create_chroma_collection(self.chroma_client, "texts_images")
        res = get_recommendation(self.s3_client, image_text_collection, text, image)
        self.after(0, lambda: self._handle_gen_result(res))


    def _browse_file_with_type(self):

        t = (self.var_file_type.get() or "Text").lower()

        if t == "text":
            patterns = [("Text (*.txt)", "*.txt")]
            allowed = {"txt"}
        elif t == "image":
            patterns = [("Image", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")]
            allowed = {"png", "jpg", "jpeg", "webp", "bmp"}

        file = filedialog.askopenfilename(
            title="Select file",
            filetypes=patterns
        )
        if not file:
            return

        ext = Path(file).suffix.lower().lstrip(".")
        if allowed and ext not in allowed:
            logger.info("File type not supported")
            return

        self.var_file_path.set(file)

    def _handle_text_result(self,result,type):

        area = self.reply
        for w in area.winfo_children():
            w.destroy()
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]
        text = ""
        for rank, (doc_id, doc, dist) in enumerate(zip(ids, docs, dists), start=1):
            temp_path = doc.split("/")[-1]
            text += (f"{rank}. id={doc_id}, distance={dist:.4f}")
            text += (f"   document: {doc}\n")
            if (type == "image"):
                continue
            text += ("\n" + get_text(self.s3_client, "trusted-zone", f"texts/{temp_path}") + "\n")
        txt = tk.Text(area, wrap="word")
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        txt.grid(row=0, column=0, sticky="nsew")

        ybar = ttk.Scrollbar(area, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=ybar.set)
        ybar.grid(row=0, column=1, sticky="ns")

        area.grid_rowconfigure(0, weight=1)
        area.grid_columnconfigure(0, weight=1)

    def _handle_multi_modal_result(self, res_text,res_img):
        area = self.reply1
        for w in area.winfo_children():
            w.destroy()

        text = "Images: \n"
        text += res_img
        text += "\n"
        text += "-"*40
        text += "\n"
        text += "Texts: \n"
        text += res_text
        txt = tk.Text(area, wrap="word")
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        txt.grid(row=0, column=0, sticky="nsew")

        ybar = ttk.Scrollbar(area, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=ybar.set)
        ybar.grid(row=0, column=1, sticky="ns")

        area.grid_rowconfigure(0, weight=1)
        area.grid_columnconfigure(0, weight=1)

    def _handle_gen_result(self,text):
        area = self.reply2
        for w in area.winfo_children():
            w.destroy()

        txt = tk.Text(area, wrap="word")
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        txt.grid(row=0, column=0, sticky="nsew")

        ybar = ttk.Scrollbar(area, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=ybar.set)
        ybar.grid(row=0, column=1, sticky="ns")

        area.grid_rowconfigure(0, weight=1)
        area.grid_columnconfigure(0, weight=1)

class TkQueueHandler(logging.Handler):
    """
    Thread-safe logging handler that enqueues formatted records
    to be polled by a Tkinter Text widget.
    """
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            self.handleError(record)

class LogViewer(tk.Toplevel):
    """
    A Toplevel window that shows logs in a Text widget.
    Call `attach_to(logger)` to start piping logs here.
    """
    def __init__(self, master=None, title="Logs"):
        super().__init__(master)
        self.title(title)
        self.geometry("900x420")

        self.text = tk.Text(self, wrap="word", state="disabled")
        self.scroll = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        # Queue for incoming log messages (from any thread)
        self._q = queue.Queue()
        self._handler = None
        self._formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                            "%Y-%m-%d %H:%M:%S")

        # Periodically poll the queue
        self._poll_job = self.after(100, self._poll_queue)

        # On close, detach handler cleanly
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def attach_to(self, logger: logging.Logger = None):
        """
        Attach a TkQueueHandler to the given logger (root if None).
        """
        if logger is None:
            logger = logging.getLogger()
        self._handler = TkQueueHandler(self._q)
        self._handler.setFormatter(self._formatter)
        logger.addHandler(self._handler)

    def _poll_queue(self):
        flushed = False
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                break
            else:
                if not flushed:
                    self.text.configure(state="normal")
                    flushed = True
                self.text.insert("end", msg + "\n")
                self.text.see("end")

        if flushed:
            self.text.configure(state="disabled")
        # schedule next poll
        self._poll_job = self.after(100, self._poll_queue)

    def _on_close(self):
        # detach handler
        if self._handler:
            try:
                logging.getLogger().removeHandler(self._handler)
            except Exception:
                pass
            self._handler = None
        if self._poll_job:
            try:
                self.after_cancel(self._poll_job)
            except Exception:
                pass
        self.destroy()

if __name__ == "__main__":

    args = parse_all_args()
    app = GUI(args)
    app.mainloop()
    """
    s3_client = connect_minio(args.ini)
    chroma_client = connect_chromaDB(args.chroma)
    """
    """
    if args.ini.run_option == "pipeline":

        landing_init(s3_client,args.landing)
        temporal_landing_init(s3_client,args.temporal)
        persistent_landing_init(s3_client)
        formatted_init(s3_client)
        formatted(s3_client)
        trusted_init(s3_client)
        trusted(s3_client)
        exploitation(chroma_client,s3_client)
        multi_modal_task_execution(chroma_client,s3_client)

    elif args.ini.run_option == "task":
        multi_modal_task_execution(chroma_client,s3_client)

    else:
        args.error(f"Unknown run-option: {args.run_option}")"""