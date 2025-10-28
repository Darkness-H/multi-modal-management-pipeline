from pathlib import Path
from PIL import Image

def fmt_bytes(n: int) -> str:
    units = ("B","KB","MB","GB","TB","PB")
    i, n = 0, float(n)
    while n >= 1024 and i < len(units)-1:
        n /= 1024; i += 1
    return f"{n:.2f} {units[i]}"

def prepare_file_for_search(file_path: str, file_type: str):
    """
    Prepare a local file for sending to a search engine depending on its type.

    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = file_type.lower()

    if file_type == "text":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content

    elif file_type == "image":
        # Read binary bytes
        img = Image.open(path).convert("RGB")
        return img

    else:
        raise ValueError(f"Unknown file type: {file_type}")