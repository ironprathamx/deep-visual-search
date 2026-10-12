
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "index" / "clip_index.npz"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VisualSearchEngine:
    def __init__(self, index_path=INDEX_PATH):
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}. "
                "Run src/build_index.py first."
            )

        self.device = get_device()

        # Load the saved image embeddings and their paths.
        with np.load(index_path, allow_pickle=False) as index:
            self.embeddings = index["embeddings"].astype(np.float32)
            self.paths = index["paths"].astype(str)

            saved_model = str(index["model_name"].item())
            saved_pretrained = str(index["pretrained"].item())

        if self.embeddings.ndim != 2:
            raise ValueError("Index embeddings must be a 2D matrix.")

        if len(self.embeddings) != len(self.paths):
            raise ValueError(
                "Number of embeddings does not match number of image paths."
            )

        if self.embeddings.shape[1] == 0:
            raise ValueError("Index embeddings cannot have zero dimensions.")

        # Prevent querying with a model different from the index model.
        if saved_model != MODEL_NAME or saved_pretrained != PRETRAINED:
            raise ValueError("Search model does not match the index model.")

        print(f"Loading CLIP on {self.device}...")

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                MODEL_NAME,
                pretrained=PRETRAINED,
            )
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        print(f"Index loaded: {len(self.paths)} images")
        print(f"Embedding dimensions: {self.embeddings.shape[1]}")

    @staticmethod
    def normalize(vector):
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector / np.maximum(norm, 1e-12)

    
    def _rank(self, query_embedding, top_k, exclude_path=None):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.embeddings.ndim != 2:
            raise ValueError("Index embeddings must be a 2D matrix.")

        query_embedding = np.asarray(query_embedding).reshape(-1)

        if query_embedding.size != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match index."
            )

        scores = self.embeddings @ query_embedding

        if exclude_path is not None:
            excluded = Path(exclude_path).expanduser().resolve()

            for i, relative_path in enumerate(self.paths):
                indexed_path = (ROOT / relative_path).resolve()

                if indexed_path == excluded:
                    scores[i] = -np.inf

        valid_indices = np.flatnonzero(np.isfinite(scores))
        k = min(top_k, len(valid_indices))

        if k == 0:
            return []

        candidate_indices = valid_indices[
            np.argpartition(scores[valid_indices], -k)[-k:]
        ]

        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        return [
            {
                "path": str(ROOT / self.paths[i]),
                "score": float(scores[i]),
            }
            for i in candidate_indices
        ]

    def search_by_text(self, text, top_k=10):
        if not text or not text.strip():
            raise ValueError("Text query cannot be empty.")

        tokens = self.tokenizer([text.strip()]).to(self.device)

        with torch.inference_mode():
            features = self.model.encode_text(tokens)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        query_embedding = features[0].cpu().numpy().astype(np.float32)

        return self._rank(query_embedding, top_k)

    def search_by_image(self, image_path, top_k=10):
        image_path = Path(image_path).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        with Image.open(image_path) as image:
            image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        query_embedding = features[0].cpu().numpy().astype(np.float32)

        return self._rank(
            query_embedding,
            top_k,
            exclude_path=image_path,
        )


if __name__ == "__main__":
    engine = VisualSearchEngine()

    print("\n--- Text-to-image search ---")
    results = engine.search_by_text("a building with distinctive architecture", top_k=5)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. Score: {result['score']:.4f} | "
            f"{result['path']}"
        )
