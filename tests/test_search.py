
from pathlib import Path

import numpy as np
import pytest

from src import search


def make_engine(embeddings=None, paths=None):
    """Create a search engine without loading CLIP."""
    engine = search.VisualSearchEngine.__new__(
        search.VisualSearchEngine
    )

    engine.embeddings = np.asarray(
        embeddings if embeddings is not None else [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    engine.paths = np.asarray(
        paths if paths is not None else [
            "dataset/ukbench/image_a.jpg",
            "dataset/ukbench/image_b.jpg",
            "dataset/ukbench/image_c.jpg",
        ]
    )

    return engine


def test_rank_returns_results_in_descending_order():
    engine = make_engine()

    results = engine._rank(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]


def test_rank_respects_top_k():
    engine = make_engine()

    results = engine._rank(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2


def test_rank_rejects_invalid_top_k():
    engine = make_engine()

    with pytest.raises(ValueError, match="top_k"):
        engine._rank(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=0,
        )


def test_rank_excludes_query_image_and_fills_results():
    engine = make_engine()

    query_path = search.ROOT / "dataset/ukbench/image_a.jpg"

    results = engine._rank(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=2,
        exclude_path=query_path,
    )

    assert len(results) == 2
    assert all(
        Path(result["path"]).resolve() != query_path.resolve()
        for result in results
    )


def test_text_search_rejects_empty_query():
    engine = make_engine()

    with pytest.raises(ValueError, match="cannot be empty"):
        engine.search_by_text("   ")


def test_image_search_rejects_missing_file():
    engine = make_engine()

    missing_path = search.ROOT / "does_not_exist.jpg"

    with pytest.raises(FileNotFoundError, match="Query image not found"):
        engine.search_by_image(missing_path)


def test_constructor_rejects_missing_index(tmp_path):
    missing_index = tmp_path / "missing.npz"

    with pytest.raises(FileNotFoundError, match="Index not found"):
        search.VisualSearchEngine(index_path=missing_index)


def test_constructor_rejects_incompatible_model_metadata(
    tmp_path, monkeypatch
):
    index_path = tmp_path / "test_index.npz"

    np.savez_compressed(
        index_path,
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        paths=np.array(["image.jpg"]),
        model_name=np.asarray("wrong-model"),
        pretrained=np.asarray(search.PRETRAINED),
    )

    def should_not_load_model(*args, **kwargs):
        pytest.fail("CLIP should not load for an incompatible index")

    monkeypatch.setattr(
        search.open_clip,
        "create_model_and_transforms",
        should_not_load_model,
    )

    with pytest.raises(ValueError, match="does not match"):
        search.VisualSearchEngine(index_path=index_path)


def test_rank_handles_top_k_larger_than_index():
    engine = make_engine()
    results = engine._rank(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=100,
    )
    assert len(results) == len(engine.paths)


def test_rank_rejects_wrong_embedding_dimension():
    engine = make_engine()

    with pytest.raises(ValueError, match="dimension"):
        engine._rank(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            top_k=3,
        )


def test_constructor_rejects_mismatched_embedding_paths(
    tmp_path, monkeypatch
):
    index_path = tmp_path / "bad_index.npz"

    np.savez_compressed(
        index_path,
        embeddings=np.zeros((2, 3), dtype=np.float32),
        paths=np.array(["image_a.jpg"]),
        model_name=np.asarray(search.MODEL_NAME),
        pretrained=np.asarray(search.PRETRAINED),
    )

    with pytest.raises(ValueError, match="does not match"):
        search.VisualSearchEngine(index_path=index_path)


def test_text_search_returns_ranked_results(monkeypatch):
    import torch

    engine = make_engine()
    engine.device = torch.device("cpu")
    engine.tokenizer = lambda texts: torch.tensor([[1]])

    class FakeModel:
        def encode_text(self, tokens):
            return torch.tensor([[1.0, 0.0]])

    engine.model = FakeModel()

    results = engine.search_by_text("a building", top_k=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert all("path" in result for result in results)
    assert all("score" in result for result in results)
