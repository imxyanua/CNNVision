from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COLOR_MODES = ("rgb", "grayscale")


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: str | Path, color_mode: str = "rgb") -> Image.Image:
    if color_mode not in COLOR_MODES:
        raise ValueError(f"Unsupported color_mode {color_mode!r}; expected one of {COLOR_MODES}")

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Not a file: {image_path}")
    # Suffix filter rejects .txt and similar before PIL tries to decode them.
    if not is_image_file(image_path):
        raise ValueError(f"Unsupported image type: {image_path}")

    try:
        image = Image.open(image_path)
        # Decode now so a truncated file fails here, not later in a transform.
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a readable image: {image_path}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read image: {image_path}") from exc

    # CNN default is 3-channel RGB. "L" is single-channel grayscale.
    if color_mode == "grayscale":
        return image.convert("L")
    return image.convert("RGB")
