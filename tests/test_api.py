
import pytest
from fastapi.testclient import TestClient

from api.main import app

from io import BytesIO
from PIL import Image

class FakeSearchEngine:
    """A lightweight substitute for the real CLIP engine."""

    def search_by_text(self, query, top_k=10):
        return [
            {"path": f"image_{i}.jpg", "score": 0.9 - i * 0.01}
            for i in range(top_k)
        ]

    def search_by_image(self, image_path, top_k=10):
        return [
            {"path": f"image_{i}.jpg", "score": 0.9 - i * 0.01}
            for i in range(top_k)
        ]


@pytest.fixture
def client(monkeypatch):
    # Replace the expensive CLIP engine with our fake engine.
    monkeypatch.setattr(
        "api.main.VisualSearchEngine",
        FakeSearchEngine,
    )

    # TestClient runs the app's startup and shutdown lifecycle.
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_text_search_success(client):
    response = client.post(
        "/search/text",
        data={"query": "a red car", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["query"] == "a red car"
    assert response.json()["count"] == 3
    assert len(response.json()["results"]) == 3

def test_invalid_image_contents(client):
    response = client.post(
        "/search/image",
        files={
            "file": (
                "fake.png",
                b"this is not an image",
                "image/png",
            )
        },
        data={"top_k": 3},
    )

    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]


def test_empty_text_query(client):
    response = client.post(
        "/search/text",
        data={"query": "   ", "top_k": 5},
    )

    assert response.status_code == 400


def test_invalid_top_k(client):
    response = client.post(
        "/search/text",
        data={"query": "a red car", "top_k": 25},
    )

    assert response.status_code == 400


def test_image_search_success(client):
    # Create a valid PNG image in memory.
    image_buffer = BytesIO()

    Image.new("RGB", (2, 2), color="red").save(
        image_buffer,
        format="PNG",
    )

    image_bytes = image_buffer.getvalue()

    # Send the image to the API.
    response = client.post(
        "/search/image",
        files={
            "file": (
                "query.png",
                image_bytes,
                "image/png",
            )
        },
        data={"top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "query.png"
    assert response.json()["count"] == 3



def test_unsupported_image_type(client):
    response = client.post(
        "/search/image",
        files={
            "file": (
                "document.txt",
                b"some text",
                "text/plain",
            )
        },
        data={"top_k": 3},
    )

    assert response.status_code == 415

def test_text_query_too_long(client):
    response = client.post(
        "/search/text",
        data={"query": "a" * 501, "top_k": 5},
    )

    assert response.status_code == 400
    assert "500 characters or fewer" in response.json()["detail"]

def test_image_exceeding_pixel_limit(client):
    from io import BytesIO

    from PIL import Image

    image_buffer = BytesIO()
    Image.new("RGB", (7000, 6000)).save(
        image_buffer,
        format="PNG",
    )

    response = client.post(
        "/search/image",
        files={
            "file": (
                "large.png",
                image_buffer.getvalue(),
                "image/png",
            )
        },
        data={"top_k": 3},
    )

    assert response.status_code == 400
    assert "40 megapixels" in response.json()["detail"]