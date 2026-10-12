
from pathlib import Path
import time

import numpy as np
import torch
import open_clip
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "ukbench"
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "clip_index.npz"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    image_paths = sorted(
        path for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No supported images found in {DATASET_DIR}")

    device = get_device()
    print(f"Images found: {len(image_paths)}")
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME} ({PRETRAINED})...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device).eval()

    embeddings = []
    valid_paths = []
    skipped = 0
    start = time.perf_counter()

    for offset in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Indexing"):
        batch_paths = image_paths[offset:offset + BATCH_SIZE]
        batch_images = []
        batch_valid_paths = []

        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    batch_images.append(preprocess(image.convert("RGB")))
                batch_valid_paths.append(path)
            except (OSError, UnidentifiedImageError) as exc:
                print(f"Skipping unreadable image {path.name}: {exc}")
                skipped += 1

        if not batch_images:
            continue

        image_tensor = torch.stack(batch_images).to(device)

        with torch.inference_mode():
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        embeddings.append(features.cpu().numpy().astype(np.float32))
        valid_paths.extend(str(path.relative_to(ROOT)) for path in batch_valid_paths)

    if not embeddings:
        raise RuntimeError("No images could be indexed.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embedding_matrix = np.concatenate(embeddings, axis=0)

    np.savez_compressed(
        INDEX_PATH,
        embeddings=embedding_matrix,
        paths=np.asarray(valid_paths),
        model_name=np.asarray(MODEL_NAME),
        pretrained=np.asarray(PRETRAINED),
    )

    elapsed = time.perf_counter() - start
    print(f"\nIndex saved: {INDEX_PATH}")
    print(f"Indexed images: {len(valid_paths)}")
    print(f"Skipped images: {skipped}")
    print(f"Embedding shape: {embedding_matrix.shape}")
    print(f"Elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
